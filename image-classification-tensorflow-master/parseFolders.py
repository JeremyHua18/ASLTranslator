import imageio
import numpy as np
import os
import string
# img = np.asarray((imageio.imread('./A1.jpg')))
# print(img.shape)

source = './datasetProcessed800/'
destination = './datasetProcessed800organized/'

#create folders A-Z

letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "space", "nothing", "del"]

for letter in letters:
    os.mkdir(destination + letter)



    # change image directory

    # for i in range(1, 301):
    #     # i = 1
    #     os.rename(source + str(al) + '/' + str(al) + str(i) + '.jpg', destination + str(al) + '/' + str(al)  + str(i) + '.jpg')



#organize random images into folder

# # print('normal_A31.jpg'[7:])
# print('processed_del2.jpg'[10:13])

alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
special = ["space", "nothing", "del"]

for filename in os.listdir(source): 
    if filename.endswith('.jpg') and not filename == '.jpg':   
        print(filename)
        if filename[10:15] == 'space':
            os.rename(source + filename, destination + filename[10:15] + '/' + filename[10:])
        elif filename[10:17] == 'nothing':
            os.rename(source + filename, destination + filename[10:17] + '/' + filename[10:])
        elif filename[10:13] == 'del':
            os.rename(source + filename, destination + filename[10:13] + '/' + filename[10:])
        else:
            os.rename(source + filename, destination + filename[10] + '/' + filename[10:])
        
        
#         print(filename)
#         break
#         # print(filename)
#         # if filename[10] in alphaList:
#         if filename[10:15] == 'space':
#             os.rename(source + str(filename), destination + 'space' + '/' + filename[10:])

            
            
            
            
            
            
            
            
            
            
            
            
            