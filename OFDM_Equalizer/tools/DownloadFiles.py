import urllib.request
import os

def Download_Mat_files():
    url      = ['https://drive.google.com/u/0/uc?id=1SaEYvjrwg6dOa_vStezJsSLnzNyaUool&export=download&confirm=t&uuid=8405fc7d-2087-4971-b6c7-aa6d8bf6163a&at=ALgDtsySNKq3_f8I_MFo8t_8V_NX:1674795448925',
            'https://drive.google.com/u/0/uc?id=1_b2y79pj58TGkEEKVOiO4vKSemoLzl7V&export=download&confirm=t&uuid=175d3890-2f06-4e3b-afc2-e4d6ed613948&at=ALgDtsy7uxj_kXYGuMH5uASQKQ-B:1674795629286']
    filename = ["/v2v80211p_LOS.mat","/v2v80211p_NLOS.mat"]
    main_path = os.path.dirname(os.path.abspath(__file__))+"/.."
    data_path = main_path+"/Data"
    if(os.path.exists(data_path) == False):
        os.mkdir(data_path)
        
    for n in range(0,2):
        file_path = data_path+filename[n]

        cwd   = os.getcwd()
        file_ = open(file_path, 'wb')

        resp = urllib.request.urlopen(url[n])
        length = resp.getheader('content-length')
        if length:
            length = int(length)
            blocksize = max(4096, length//100)
        else:
            blocksize = 1000000 # just made something up

        #print(length, blocksize)

        size = 0
        it = 0
        while True:
            buf1 = resp.read(blocksize)
            if not buf1:
                break
            file_.write(buf1)
            size += len(buf1)
            it+=1
            if it%5 == 0:
                print('{} {}%\r Download: '.format(filename[n],int((size/length)*100)), end='')
        print()
        file_.close()
