import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from set_transformer import SetTransformer
from tqdm import tqdm


def train_epoch(model, data, optimizer, args):
    """ Epoch operation in training phase. """
    model.train()

    #print(data.shape)
    total_elements = data.shape[0]
    
    # Compute how many data points we can fit
    num_used_elements = (total_elements // (args.batch_size * args.set_size)) * (args.batch_size * args.set_size)
    
    # Truncate data
    truncated_data = data[:num_used_elements]
    
    N = truncated_data.shape[0] // (args.batch_size * args.set_size)
    
    data = truncated_data.view(N, args.batch_size * args.set_size, -1)
    #print(reshaped_data.shape)
    N, _, dim = data.shape
    
    # Randomly shuffle the second dimension
    idx = torch.randperm(args.batch_size * args.set_size)
    data = data[:, idx, :]
    #print(data.shape)
    data = data.view(N, args.batch_size, args.set_size, dim)
    #print(data.shape)
    

    positive_loss = 0

    negative_loss = 0

    for batch in tqdm(data, mininterval=2, desc='  - (Training)   ', leave=False):
        """ forward """
        #print(batch.shape)
        optimizer.zero_grad()
        set_embedding = model(batch)  # batch_size*dim
        #print(batch.shape)

        # Constructing negative samples
        batch_size, seq_len, dim = batch.shape
        selected_indices = torch.randint(0, seq_len, (batch_size, args.fake_num))

        # 为每个选择的索引随机选择操作
        factors = torch.where(torch.rand(batch_size, args.fake_num) < 0.5, args.negative_parameter, 1/args.negative_parameter)
        #print(factors.shape)

        # 通过广播，创建一个形状为 [batch_size, k, dim] 的张量来存储选择的操作
        factors = factors.unsqueeze(-1)

        # 使用高级索引将选择的操作应用到原始张量上
        batch_negative = batch.scatter_add_(1, selected_indices.unsqueeze(-1).expand(-1, -1, dim),
                            (factors - 1) * batch.gather(1, selected_indices.unsqueeze(-1).expand(-1, -1, dim)))

        neg_embedding = model(batch_negative)


        
        """ backward """
        difference_positive = set_embedding.unsqueeze(1) - set_embedding.unsqueeze(0)
        difference_negative = neg_embedding.unsqueeze(1) - neg_embedding.unsqueeze(0)
        loss1 = torch.norm(difference_positive, p=2)
        loss2 = torch.norm(difference_negative, p=2)
        loss = loss1-loss2*args.scale_negative
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        positive_loss += loss1
        negative_loss += loss2
    
    return positive_loss, negative_loss


def eval_epoch(model, data, args):
    """ Epoch operation in eval phase. """
    #model.train()
    # print(data.shape)
    total_elements = data.shape[0]

    # Compute how many data points we can fit

    num_used_elements = (total_elements // (args.batch_size * args.set_size)) * (args.batch_size * args.set_size)

    # Truncate data

    truncated_data = data[:num_used_elements]
    N = truncated_data.shape[0] // (args.batch_size * args.set_size)
    data = truncated_data.view(N, args.batch_size * args.set_size, -1)  # 这个data是没有打乱的   N*16*15*8


    # step

    step = N // args.windows_num

    # selecting by step

    data = data[::step][:args.windows_num]
    #print(data.shape)
    data = data.view(args.windows_num, args.batch_size, args.set_size, -1)

    tensor_list = []
    with torch.no_grad():
        for batch in tqdm(data, mininterval=2, desc='  - (Evaluating)   ', leave=False):
            """ forward """
            #print(batch.shape)
            set_embedding = model(batch)  # batch_size*dim
            tensor_list.append(set_embedding)

    stacked_tensor = torch.stack(tensor_list)
    # step 1: mean
    mean_tensor = torch.mean(stacked_tensor, dim=0)
    #assert mean_tensor.shape == (16, 8)

    # step: pairwise square
    diff = mean_tensor.unsqueeze(1) - mean_tensor.unsqueeze(0)
    square_diff = diff ** 2
    result_tensor = square_diff.sum(dim=2)

    #assert result_tensor.shape == (16, 16)
    #print(result_tensor)

    return result_tensor



def train(model, data, optimizer, scheduler, args):
    """ Start training. """
    for epoch_i in range(args.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        #start = time.time()
        #train_data = 
        positive_loss, negative_loss = train_epoch(model, data, optimizer, args)
        print('  - (Training)    positive loss: {ll: 8.5f}, '
              .format(ll=positive_loss))
        print('  - (Training)    negative loss: {ll: 8.5f}, '
              .format(ll=negative_loss))

        eval_matrix = eval_epoch(model, data, args)
        print('  - (Evaluating)    eval matrix:')
        print(eval_matrix)

        # # logging
        # with open(opt.log, 'a') as f:
        #     f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}\n'
        #             .format(epoch=epoch, ll=valid_event, acc=valid_type))

        scheduler.step()





def main():
    """ Main function. """

    parser = argparse.ArgumentParser(description="Your model description here.")

    # data
    parser.add_argument('--data-path', default='data/NEweather_data.csv', type=str, help='Path to data.')


    # hyper parameters
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--set_size', default=15, type=int)
    parser.add_argument('--n_layers', default=2, type=int)
    parser.add_argument('--n_head', default=4, type=int)
    parser.add_argument('--input_dim', default=8, type=int)
    parser.add_argument('--fake_num', default=6, type=int)  # the number of fake samples in each negative set
    parser.add_argument('--scale_negative', default=00.1, type=int)  # negative loss is large and need to be scaled
    parser.add_argument('--negative_parameter', default=1.3, type=int)
    parser.add_argument('--windows_num', default=10, type=int)  # the number of windows


    

    args = parser.parse_args()

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    print(f'[Info] Using device: {device}')
    print(f'[Info] Loaded parameters: {args}')

    # load data
    data = load_data(args.data_path)

    # input_dim = data.shape[1]
    # print(input_dim)
    # parser.add_argument('--input_dim', default=input_dim, type=int)
    



    # model initialize
    model = SetTransformer(n_heads=args.n_head, n_layers=args.n_layers, dim=args.input_dim)
    model.to(device)

    
    #optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           args.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    
    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, data, optimizer, scheduler, args)
    #eval(model, data, optimizer, scheduler, args)
    #def train(model, data, optimizer, scheduler, args):


def load_data(data_path):
    df = pd.read_csv(data_path)
    numpy_data = df.values
    torch_tensor = torch.from_numpy(numpy_data)
    return torch_tensor


# def train_model():
#     """ Virtual training function. """
#     pass


if __name__ == "__main__":
    main()
