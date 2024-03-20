import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, time, random, argparse
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

import warnings
warnings.filterwarnings(action='ignore')


"""1. Dataset 준비"""
"""1-1. Data 전처리"""
class HAR_preprocess():
    def __init__(self, configs):
        self.configs = configs

        # load x data
        train_data = self.make_data(glob(os.path.join(self.configs.data_dir, 'train/Inertial Signals/*')))
        test_data = self.make_data(glob(os.path.join(self.configs.data_dir, 'test/Inertial Signals/*')))

        # load y data
        train_label = (pd.read_csv(os.path.join(self.configs.data_dir, 'train/y_train.txt'), 
                                    header=None, sep=' ').values-1).squeeze()
        test_label = (pd.read_csv(os.path.join(self.configs.data_dir, 'test/y_test.txt'), 
                                    header=None, sep=' ').values-1).squeeze()

        # split train/valid/test
        train_data, train_label, valid_data, valid_label = self.split_train_valid(train_data, train_label, train_ratio=0.8)

        self.trainset = {'X': train_data, 'y': train_label}
        self.validset = {'X': valid_data, 'y': valid_label}
        self.testset = {'X': test_data, 'y': test_label}

    @staticmethod
    def split_train_valid(x, y, train_ratio:float=0.8):
        len_train = int(len(x)*train_ratio)

        train_data, train_label = x[:len_train,:,:], y[:len_train]
        valid_data, valid_label = x[len_train:,:,:], y[len_train:]

        return train_data, train_label, valid_data, valid_label

    def make_data(self, data_path):
        data = []
        for i in range(len(data_path)):
            data.append(self.create_data(data_path[i]))
        data = np.transpose(np.array(data), (1,0,2))
        return data
    
    @staticmethod
    def create_data(path_name):
        with open(path_name) as f:  # os.path.join(path, filename) 경로의 file을 f 라는 이름으로 open
            data = []
            for line in f:  # f 라는 이름의 txt 데이터를 line 별로 for문을 돌면서 읽기
                num = [float(l) for l in line.split()]  # f라는 이름의 file 내의 line을 for문을 돌며 float 형태로 변환
                data.append(num)  # 각 값을 data로 만들기 위해 list 안에 담기
        data = np.array(data).reshape(-1,128)  # list 형태의 데이터를 np.array 형태로 바꾸어주기
        return data
    
    def jitter(self, x, sigma=0.001):
        noise = np.random.normal(loc=0, scale=sigma, size=x.shape)
        return x + noise

    def scaling(self, x, sigma=0.2):
        scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(1, x.shape[1])) # shape=(1,3)
        noise = np.matmul(np.ones((x.shape[0],1)), scaling_factor)
        return x*noise

"""1-2. Custom Dataset 정의"""
class HARDataset(Dataset):
    def __init__(self, configs, mode:str):
        self.configs = configs
        self.get_data = HAR_preprocess(self.configs)

        if mode == "train":
            self.X, self.y = torch.Tensor(self.get_data.trainset['X']), torch.Tensor(self.get_data.trainset['y'])
        elif mode == "valid":
            self.X, self.y = torch.Tensor(self.get_data.validset['X']), torch.Tensor(self.get_data.validset['y'])
        elif mode == "test":
            self.X, self.y = torch.Tensor(self.get_data.testset['X']), torch.Tensor(self.get_data.testset['y'])
        else:
            raise ValueError

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        # Transformation
        rand_prob = np.random.uniform()
        if self.configs.jitter and rand_prob > self.configs.aug_prob:
            X = self.get_data.jitter(X)
        if self.configs.scale and rand_prob > self.configs.aug_prob:
            X = self.get_data.scaling(X)

        return {'X':X.float(), 'y':y}

"""1-3. Dataloader 호출 함수화"""
def get_dataloader(configs):
    trainset = HARDataset(configs, mode="train")
    validset = HARDataset(configs, mode="valid")
    testset = HARDataset(configs, mode="test")

    trainloader = DataLoader(trainset, batch_size=configs.batch_size, shuffle=True, drop_last=False)
    validloader = DataLoader(validset, batch_size=configs.batch_size, shuffle=False, drop_last=False)
    testloader = DataLoader(testset, batch_size=configs.batch_size, shuffle=False, drop_last=False)

    return trainloader, validloader, testloader


"""2. 모델 정의"""
class CNN1d(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.seq_len = configs.seq_len
        self.num_classes = configs.num_classes
        self.in_channels = configs.in_channels
        self.mid_channels = configs.mid_channels
        self.out_channels = configs.out_channels
        self.kernel_size = configs.kernel_size
        self.stride = configs.stride
        self.p = configs.dropout_rate

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.mid_channels,
                        self.kernel_size, self.stride),
            nn.BatchNorm1d(self.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(self.p),
        )
        # calculate sequence length (L_out)
        L_out = self.calculate_output_length(self.seq_len, self.kernel_size, self.stride)//2
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.mid_channels, self.out_channels,
                        self.kernel_size, self.stride),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(self.p),
        )
        # calculate sequence length (L_out)
        L_out = self.calculate_output_length(L_out, self.kernel_size, self.stride)//2

        self.classifier = nn.Linear(self.out_channels*L_out, self.num_classes)
    
    @staticmethod
    def calculate_output_length(input_seq, kernel_size, stride, padding=0, dilation=1):
        return (input_seq + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def forward(self,x):
        out = self.conv1(x) 
        out = self.conv2(out)   # (B, out_channels, L_out)
        out = out.view(out.size(0), -1)  # flatten: (B, out_channels, L_out) -> (B, out_channels * L_out)
        out = self.classifier(out)  # (B, num_classes)
        
        return out

"""3. Optimizer 정의 함수화"""
def get_optimizer(configs, model):
    if configs.optimizer == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=configs.lr,
                              momentum=configs.momentum, weight_decay=configs.weight_decay)

    elif configs.optimizer == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=configs.lr, 
                                    betas=(configs.momentum, 0.999), # (momentum, adaptive lr momentum)
                                    weight_decay=configs.weight_decay)
    else:
        print("Invalid optimizer name, it should be one of ['sgd', 'adam']. Exiting...")
        exit()

    return optimizer


"""4. Train 과정 함수화"""
def train(device, model, trainloader, criterion, optimizer):
    num_iterations = len(trainloader)
    train_history = {'loss': torch.zeros(num_iterations),      # 모델의 손실을 기록하기 위한 변수
                    'accuracy': torch.zeros(num_iterations)}  # 모델의 성능(accuracy)을 기록하기 위한 변수
    
    for i, batch in enumerate(trainloader):
        # X, y 데이터 명시, 두 텐서를 모델, 목적함수와 같은 device로
        X, y = batch['X'].to(device), batch['y'].to(device).long()

        # forward
        logits = model(X)  # model에 input_imgs를 입력으로 넣으면 자동으로 forward 함수가 호출되어 prediction을 output으로 도출

        # Loss 계산!
        loss = criterion(logits, y)

        """계산된 loss에서 gradient를 계산하는 역전파 함수: .backward()"""
        loss.backward()  # PyTorch 자동 미분

        """optimizer를 이용한 파라미터 업데이트"""
        optimizer.step()
        
        """Optimizer Gradient 초기화"""
        optimizer.zero_grad()

        # 모델 성능 계산!
        max_pred = torch.max(logits.detach(), dim=-1)[1] # 샘플 별 logit의 max 값들을 뽑기 -> [1]로 max 값들의 idx를 반환
                                                         # logit의 max 값은 모델이 해당 class로 예측했다는 의미
        accuracy = torch.eq(max_pred, y).sum().cpu().item() / len(y)  # 예측한 class와 정답이 얼마나 맞는지(eq) 비교하고, 맞은 개수를 합한 후 (sum) 평균 내기

        train_history['loss'][i] = loss.item()   # 'item()'은 tensor의 item 값 (상수 값)만 반환. tensor를 직접적으로 리스트에 append하면 불필요하게 메모리가 쌓이는 것 주의.
        train_history['accuracy'][i] = accuracy

    train_result = {k: v.mean().item() for k, v in train_history.items()}  # train의 loss, acc를 평균내어 결과 보기 

    return train_result

"""5. Valid 과정 함수화"""
def evaluate(device, model, validloader, criterion):
    model.eval() # batchnorm, dropout 등 train할 때와 test할 때 연산이 다른 경우가 존재

    num_iterations = len(validloader)
    valid_history = {'loss': torch.zeros(num_iterations),      # 모델의 손실(loss)을 기록하기 위함
                    'accuracy': torch.zeros(num_iterations)}   # 모델의 성능(accuracy)을 기록하기 위함

    with torch.no_grad():  # 자동미분 연산 중지
        for i, batch in enumerate(validloader):
            X, y = batch['X'].to(device), batch['y'].to(device).long()
            logits = model(X)

            loss = criterion(logits, y)

            """loss를 backward 하지 않는 것에 주의"""

            # 모델 손실 및 성능 기록
            max_pred = torch.max(logits.detach(), dim=-1)[1]
            accuracy = torch.eq(max_pred, y).sum().cpu().item() / len(y)  # 예측한 class와 정답이 얼마나 맞는지(eq) 비교하고, 맞은 개수를 합한 후 (sum) 평균 내기

            valid_history['loss'][i] = loss.item()   # 모델 손실 저장
            valid_history['accuracy'][i] = accuracy  # 모델 성능 저장

    valid_result = {k: v.mean().item() for k, v in valid_history.items()}

    return valid_result

"""6. 학습 추세 시각화 함수화"""
def visualize_model_training(configs, epoch_history):
    # Loss 추세 시각화
    sns.lineplot(x=range(1, configs.num_epochs+1), y=epoch_history['train_loss'], label="Train Loss")
    sns.lineplot(x=range(1, configs.num_epochs+1), y=epoch_history['valid_loss'], label="Valid Loss")
    plt.title("Train vs Valid Loss Graph")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="best")
    plt.savefig(os.path.join(configs.save_dir, "train_valid_loss.png"), dpi=350)
    plt.close('all')

    # 성능(Accuracy) 추세 시각화
    sns.lineplot(x=range(1, configs.num_epochs+1), y=epoch_history['train_acc'], label="Train Accuracy")
    sns.lineplot(x=range(1, configs.num_epochs+1), y=epoch_history['valid_acc'], label="Valid Accuracy")
    plt.title("Train vs Valid Accuracy Graph")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    plt.savefig(os.path.join(configs.save_dir, "train_valid_acc.png"), dpi=350)
    plt.close('all')

"""7. 모델 파라미터 저장 함수화"""
def save_checkpoint(save_dir, model):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    save_model_dir = os.path.join(save_dir, 'HAR_model_checkpoint.pt')

    save_dict = {'model': model.state_dict()}
    torch.save(save_dict, save_model_dir)

"""8. Test 과정 함수화"""
def test(device, configs, model, testloader):
    """test model 정의"""
    saved_model = torch.load(os.path.join(configs.save_dir, 'HAR_model_checkpoint.pt')) # 모델 불러오기
    parameters = saved_model['model']  # 'model'의 파라미터 불러오기
    model.load_state_dict(parameters)

    """test 과정에서 pred, true를 모두 반환"""
    test_result = {'pred': [], 'true': []}

    model.eval()
    with torch.no_grad():
        for batch in testloader:
            X, y = batch['X'].to(device), batch['y'].to(device).long()
            
            logits = model(X) # B,2
            # 출력 값을 최대로 하는 인덱스(class 저장)
            pred = torch.argmax(logits, dim=1)

            test_result['pred'].extend(pred.squeeze().cpu().numpy())
            test_result['true'].extend(y.squeeze().cpu().numpy())

    return test_result


"""9. 모든 학습 프레임워크 함수화: `main()`"""
def main(configs):
    """Step 1: 학습 전 세팅: device, seed, configs 확인, data 정의, model 정의, loss function 및 optimizer 정의"""
    # Device 정의
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # seed 정의
    torch.manual_seed(configs.seed) # torch cpu seed 고정
    torch.cuda.manual_seed(configs.seed) # torch gpu seed 고정
    torch.cuda.manual_seed_all(configs.seed) # torch multi-gpu seed 고정
    np.random.seed(configs.seed) # numpy seed 고정
    random.seed(configs.seed) # python seed 고정

    # 우리가 설정한 하이퍼파라미터가 무엇인지 프린트하여 확인
    print(f"Configurations: \n {configs}")

    # 데이터 불러오기
    trainloader, validloader, testloader = get_dataloader(configs)
    
    # 모델 정의하기
    model = CNN1d(configs)
    # GPU로 모델 및 loss function 전송 (CPU에서 계산하는 경우 연산량에 따라 시간이 오래 걸림)
    model, criterion = model.to(device), nn.CrossEntropyLoss().to(device)

    # optimizer 정의
    optimizer = get_optimizer(configs, model)  # 앞서 정의해두었던 함수 호출

    # training
    best_acc = 0.0              # 최종적으로 저장할 모델을 판단할 기준 설정 (초기화)
    best_epoch = 0.0            # 최종 모델이 몇 번째 epoch에서 도출되는지 저장 위함 (초기화)
    epoch_history = {'train_loss': [], 'train_acc': [],
                     'valid_loss': [], 'valid_acc': []}

    start = time.time()

    """Step 2: 학습 시작"""
    for epoch in range(1, configs.num_epochs+1):  

        print('-'*15, f'Epoch: {epoch}', '-'*15)

        train_result = train(device, model, trainloader, criterion, optimizer)     # 앞서 정의해두었던 함수 호출
        valid_result = evaluate(device, model, validloader, criterion)  # 앞서 정의해두었던 함수 호출

        print(f"Train Loss: {train_result['loss']:.2f} | Train Accuracy: {train_result['accuracy']:.2f}")
        print(f"Valid Loss: {valid_result['loss']:.2f} | Valid Accuracy: {valid_result['accuracy']:.2f}")

        """Step 3: 학습 중 도출된 loss 및 acc 값들 저장"""
        epoch_history['train_loss'].append(train_result['loss'])
        epoch_history['train_acc'].append(train_result['accuracy'])
        epoch_history['valid_loss'].append(valid_result['loss'])
        epoch_history['valid_acc'].append(valid_result['accuracy'])

        """Step 4: 모델을 평가 (valid/eval) 하였을 때 성능 측면에서 가장 우수했던 모델을 저장"""
        if valid_result['accuracy'] > best_acc:
            best_acc = valid_result['accuracy']  # best acc 값 업데이트
            best_epoch = epoch                   # best epoch 값 업데이트
            save_checkpoint(configs.save_dir, model)  # 앞서 정의해두었던 함수 호출

    # 최종적으로 선택된 모델에 대한 값 확인
    print(f"Best Valid Accuracy:{best_acc:.2f} | Best Epoch:{best_epoch}")

    # 학습 추세 시각화
    if configs.visualize:
        visualize_model_training(configs, epoch_history)

    """Step 5: 모델 최종 성능 평가"""
    test_result = test(device, configs, model, testloader)
    # print(f"Test Accuracy  : {test_result['accuracy']:.2f} \n")

    # 학습/테스트에 소요된 시간 계산 후 출력
    end_sec = time.time() - start
    end_min = end_sec / 60
    print(f"Total Training Time: {end_min:.2f} minutes")

    # Test 결과 확인
    true, pred = test_result['true'], test_result['pred']
    conf_mat = confusion_matrix(test_result['true'], test_result['pred'])

    # confusion matrix 시각화
    plt.figure(figsize=(10,8))
    sns.heatmap(data=conf_mat, annot=True, fmt='d', annot_kws={'size':15}, cmap="Blues")
    plt.xlabel('Predicted', size=20)
    plt.ylabel('True', size=20)
    plt.savefig(os.path.join(configs.save_dir, "confusion_matrix.png"), dpi=350)
    plt.close('all')

    # 평가지표 계산
    test_score = dict()
    test_score['test_acc'] = accuracy_score(true, pred)
    test_score['test_recall'] = recall_score(true, pred, average='macro')
    test_score['test_precision'] = precision_score(true, pred, average='macro')
    test_score['test_f1'] = f1_score(true, pred, average='macro')

    print('Test Accuracy   : {:.3f}'.format(test_score['test_acc']))
    print('Test Sensitivity: {:.3f}'.format(test_score['test_recall']))
    print('Test Precision  : {:.3f}'.format(test_score['test_precision']))
    print('Test F1 Score   : {:.3f}'.format(test_score['test_f1']))

    return test_score


"""10. main 함수 실행"""
if __name__ == '__main__':

    """11. 실험에 필요한 hyperparameters 정의"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    
    # data 관련 hyperparameters
    parser.add_argument("--data-dir", default='./data/UCI HAR Dataset/', type=str)
    parser.add_argument("--jitter", default=False, type=bool)
    parser.add_argument("--scale", default=False, type=bool)
    parser.add_argument("--aug-prob", default=.5, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--in-channels", default=9, type=int)
    parser.add_argument("--seq-len", default=128, type=int)
    parser.add_argument("--num-classes", default=6, type=int)
    
    # model 관련 hyperparameters
    parser.add_argument("--mid-channels", default=196, type=int)
    parser.add_argument("--out-channels", default=64, type=int)
    parser.add_argument("--kernel-size", default=6, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--dropout-rate", default=.1, type=float)

    # train 관련 hyperparameters
    parser.add_argument("--optimizer", default="adam", type=str, choices=["adam", "sgd"])
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--momentum", default=.9, type=float)
    parser.add_argument("--weight-decay", default=.0, type=float)
    parser.add_argument("--num-epochs", default=15, type=int)
    parser.add_argument("--save-dir", default="./results_HAR/", type=str)
    parser.add_argument("--visualize", default=True, type=bool)

    configs = parser.parse_args()

    main(configs)  # main 함수 실행