import urllib.request
import os

def Download_Mat_files():
    url      = ['https://drive.google.com/u/0/uc?id=1UhQhNHwPk5Ak8m8zKcE1z20eyu_9wYft&export=download&confirm=t&uuid=7ea3f81a-9d62-4a00-8784-e15ba740a2c9&at=ALgDtsyZmhpQ19Kpv5xArwfAEUBG:1675459749342',
                'https://drive.google.com/u/0/uc?id=1aIN733FtAGO1BGg_5aO-hEHy4KValrNR&export=download&confirm=t&uuid=cfcae551-bf79-4b24-a949-4aaee5fb89f5&at=ALgDtswchvQSQyHttK7ADc33Ad-p:1675459810282']
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
