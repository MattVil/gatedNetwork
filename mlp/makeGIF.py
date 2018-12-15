import glob
import imageio

filenames = [img for img in glob.glob("GIF/*.png")]
filenames.sort()

images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('evol.gif', images)
