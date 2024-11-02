import torch
import numpy as np
from PIL import Image
import random
import os
from torch.nn.functional import one_hot
from model import final_model
from torch.optim import Adam

def load_data(path, size=[92, 38]):
    image = Image.open(path).resize(size).convert('L')  # 转换为灰度图
    image = image.point(lambda x: 0 if x < 255 else 255, '1')
    return torch.from_numpy(np.array(image)).float().unsqueeze(0)

def get_batch(num, main_path='./data'):
    path = random.choices(os.listdir(main_path), k=num)
    label = [*map(lambda x: x.split('.')[0], path)]
    label = [*map(lambda x: x.split('_')[0], label)]
    label = [*map(lambda x: [int(i) for i in x], label)]
    label = one_hot(torch.Tensor(label).long(), 10)
    path = [*map(lambda x: main_path+'/'+x, path)]
    feature = torch.stack([load_data(i) for i in path])
    return feature, label

if __name__ == '__main__':
    model = final_model()
    opt = Adam(model.parameters(), lr=0.0001)

    for i in range(1000000):
        model.train()
        opt.zero_grad()
        x, y = get_batch(120)
        ypre = model(x)
        loss = -torch.mean(torch.log(torch.sum(ypre * y, dim=-1)))
        loss.backward()
        opt.step()

        # 测试集验证
        model.eval()
        xtest, ytest = get_batch(10, './test_data')
        test_ypre = model(xtest)
        test_loss = -torch.mean(torch.log(torch.sum(test_ypre * ytest, dim=-1)))
        pred = torch.argmax(test_ypre, dim=-1)
        real = torch.argmax(ytest, dim=-1)
        equality = torch.eq(pred, real).int().sum(dim=-1) / 4
        equality = torch.where(equality < 1, 0, 1)
        acc = round(float(equality.sum() / len(equality)), 2)
        test_loss = round(float(test_loss), 2)

        print(f'Epoch[{i+0}]|loss: {round(float(loss), 2)}|test loss: {test_loss}|accuracy: {acc}')

        if i % 10 == 0:
            torch.save(model.state_dict(), '验证码识别.pth')
            model.load_state_dict(torch.load('验证码识别.pth', weights_only=True))
            xtest, ytest = get_batch(1, './test_data')
            test_ypre = model.predict(xtest)
            real = ''.join([str(i) for i in list(torch.argmax(ytest, dim=-1)[0].detach().numpy())])
            print('pre_label:', test_ypre, ' | ', 'real_label', real)
