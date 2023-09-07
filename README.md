<<<<<<< HEAD
# BotMoE Reposiotry

This is the official repo for [BotMoE: Twitter Bot Detection with Community-Aware Mixtures of Modal-Specific Experts](https://arxiv.org/pdf/2304.06280.pdf3) @ SIGIR 2023.

## Step 0 Create Env

```
    pip install -r requirements.txt
```

## Step 1 Train

Remember to change the parameters in main.py 

```
    #the name of your experiment
    exp_name="load&fix"

    #the idx of the model in the list
    idx=0

    #models
    model=[AllInOne1_rgcn_rgt_gcn]

    #your log file to record your training
    file =['AllInOne1_rgcn_rgt_gcn.log']

    #set logger
    logger=set_logger(file[idx],exp_name)

    #path to save your model
    save_root='/data3/whr/lyh/MoE/mixture-of-experts/twibot-20/model/'
    save_pth=save_root+file[idx].rstrip('.log')+'/'
    if(not os.path.exists(save_pth)):
        os.mkdir(save_pth)

    logger.info(exp_name)
    
    #the path of preprocessed features
    root='MoE/mixture-of-experts/BotRGCN/twibot_20/processed_data/'

    #hyper parameters of the model
    align_size_set=[128]
    hidden_size_set=[4]
    hidden_size=4
    device="cuda:2"
    dataset=Twibot22(root=root,device=device)
    test_run=range(20)
    num_text=2
    gnn_k=1
    num_gnn=3

```

Then you can start training!
```
    python main.py
```

## Step 2 Test your model

Change the path to your trained model

```
    trainer.model = torch.load([path/to/your/model])
```

Then run
```
    python test.py
```
