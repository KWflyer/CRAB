import torch

def select_device(id):
    force_cpu = False
    if id == -1:
        force_cpu = True
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device("cuda:{}".format(id) if cuda else "cpu")

    if not cuda:
        print("Using CPU")
    if cuda:
        print("using GPU:{}".format(id))
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        print(
            "Using CUDA device0 _CudaDeviceProperties(name='%s', total_memory=%dMB)"
            % (x[0].name, x[0].total_memory / c)
        )
        if ng > 0:
            # torch.cuda.set_device(0)  # OPTIONAL: Set GPU ID
            for i in range(1, ng):
                print(
                    "           device%g _CudaDeviceProperties(name='%s', total_memory=%dMB)"
                    % (i, x[i].name, x[i].total_memory / c)
                )

    return device

