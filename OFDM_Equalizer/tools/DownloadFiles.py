import urllib.request
import os

def Download_Mat_files():
    url      = ['https://drive.google.com/u/0/uc?id=1AF-7l6CoYaSDMyvddDfTl9jHznhQnw0-&export=download&confirm=t&uuid=b04f4665-ae21-494e-beef-f4df85eaccce&at=AHV7M3eEOBqQ7o48YX-mY-75Dqlu:1670375964900',
            'https://drive.google.com/u/0/uc?id=1xsItsz-3kn7fOoLwMlCGYjfo_RDRFH5I&export=download&confirm=t&uuid=62abbfd7-edb1-405b-902a-15e39e84ce26&at=AHV7M3daB9x4EjpqlW43xCED29ck:1670379225593']
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
