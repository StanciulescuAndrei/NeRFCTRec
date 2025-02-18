import imageio
import numpy as np    

#Create reader object for the gif
gif1 = imageio.get_reader('media/tv000.gif')
gif2 = imageio.get_reader('media/tv0025.gif')
gif3 = imageio.get_reader('media/tv005.gif')
gif4 = imageio.get_reader('media/tv01.gif')

#If they don't have the same number of frame take the shorter
number_of_frames1 = gif1.get_length()
number_of_frames2 = gif2.get_length()
number_of_frames3 = gif3.get_length()
number_of_frames4 = gif4.get_length()

#Create writer object
# new_gif = imageio.get_writer('output.gif')

images = []

for frame_number in range(number_of_frames1-1):
    img1 = gif1.get_next_data()
    img2 = gif2.get_next_data()
    img3 = gif3.get_next_data()
    img4 = gif4.get_next_data()
    #here is the magic
    new_image = np.hstack((img1, img2, img3, img4))
    images.append(new_image)
    # new_gif.append_data(new_image)

imageio.mimsave("output.gif", images, fps=3, loop=0)

gif1.close()
gif2.close()    
gif3.close()
gif4.close()    