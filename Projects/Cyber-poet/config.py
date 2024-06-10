class Config():
    data_path = '../../Dataset/'
    filename = 'tang.npz'
    use_gpu = False
    batch_size = 128
    lr = 1e-3
    save_path = 'caches/'
    max_gen_len = 200
    model_path = None
    epoch = 100