import imageio
import numpy as np
import os
import string
# img = np.asarray((imageio.imread('./A1.jpg')))
# print(img.shape)

source = './dataset/data301_2400/'
destination = './dataset/'

#create folders A-Z

# alphaList = list(string.ascii_uppercase)
# alphaList = list(string.ascii_uppercase)[3:]
alphaList = ['nothing', 'del', 'space']
# print(alphaList)
for al in alphaList:
    # os.mkdir('./dataset/' + str(al))



    # change image directory

    for i in range(1, 301):
        # i = 1
        os.rename(source + str(al) + '/' + str(al) + str(i) + '.jpg', destination + str(al) + '/' + str(al)  + str(i) + '.jpg')



#organize random images into folder

# # print('normal_A31.jpg'[7:])
# for filename in os.listdir(source): 
#     if filename.endswith('.jpg') and not filename == '.jpg':   
#         # print(filename)
#         os.rename(source + str(filename), destination + filename[7] + '/' + filename[7:])
