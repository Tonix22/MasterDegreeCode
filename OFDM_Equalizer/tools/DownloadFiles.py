import urllib.request
import os

def Download_Mat_files():
    url      = ['https://drive.google.com/u/0/uc?id=1sh4tXXERFQNsVAg6tqdNTsWax32UydAZ&export=download&confirm=t&uuid=ad2095b5-7e70-45b6-9149-eaf5b56ebce6&at=ALgDtswqDgWWAD3nLqDRiWXM1E7-:1674867736336',
                'https://drive.google.com/u/0/uc?id=1RiHYtLoFAhiEBYysA5uFJ4lEU44vnpxB&export=download&confirm=t&uuid=a276fe70-2a61-44f2-8d26-a76134f1f59b&at=ALgDtszvZuLnpTZAxWTdMS1it1iw:1674868050671']
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
