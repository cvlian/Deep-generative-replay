import h5py
from utils import *

from sklearn.preprocessing import OneHotEncoder

class Dataset:

    def getTask(self):
        return type(self).__name__, self.x, self.y

    def showSamples(self, nrows, ncols):
        """
        Plot nrows x ncols images
        """
        fig, axes = plt.subplots(nrows, ncols)
        
        for i, ax in enumerate(axes.flat):
            img = recover(self.x[i,:])
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(np.argmax(self.y[i]))
        
        plt.show()

        
class MNISTdata(Dataset):
    """
    MNIST dataset
    
    A large collection of monochrome images of handwritten digits
    
    It has a training set of 55,000 examples, and a test set of 10,000 examples
    """

    def __init__(self):
        (x_train, y_train), (x_val, y_val) = datasets.mnist.load_data()
        x_train = normalize(np.reshape(x_train[5000:], [-1, 28, 28, 1]))
        x_val = normalize(np.reshape(x_val, [-1, 28, 28, 1]))
        x_train = np.concatenate([x_train, x_train, x_train], 3)
        x_val = np.concatenate([x_val, x_val, x_val], 3)
        
        print("MNIST : Training Set", x_train.shape)
        print("MNIST : Test Set", x_val.shape)
        
        # Calculate the total number of images
        num_images = x_train.shape[0] + x_val.shape[0]
        print("MNIST : Total Number of Images", num_images)
        
        y_train = np.eye(10)[y_train[5000:]]
        y_val = np.eye(10)[y_val]
        
        self.x = np.vstack([x_train, x_val])
        self.y = np.vstack([y_train, y_val])


class USPSdata(Dataset):
    """
    USPS dataset
    
    A digit dataset automatically scanned from envelopes by the U.S. Postal Service.
    
    It contains a total of 9,298 16×16 pixel grayscale samples
    """

    def __init__(self):
        global datapath
        
        src = os.path.join(datapath, "usps.h5")
        
        with h5py.File(src, 'r') as hf:
            train = hf.get('train')
            test = hf.get('test')
            x_train = np.reshape(train.get('data')[:], [-1, 16, 16, 1])
            x_val = np.reshape(test.get('data')[:], [-1, 16, 16, 1])

            # Magnify the original images
            x_train = tf.image.resize(x_train, [28, 28])
            x_val = tf.image.resize(x_val, [28, 28])
            x_train = np.concatenate([x_train, x_train, x_train], 3)
            x_val = np.concatenate([x_val, x_val, x_val], 3)

            y_train = np.eye(10)[train.get('target')[:]]
            y_val = np.eye(10)[test.get('target')[:]]
        
            print("USPS : Training Set", x_train.shape)
            print("USPS : Test Set", x_val.shape)
        
            # Calculate the total number of images
            num_images = x_train.shape[0] + x_val.shape[0]
            print("USPS : Total Number of Images", num_images)
        
            self.x = 2*(np.vstack([x_train, x_val])-0.5)
            self.y = np.vstack([y_train, y_val])

            
class SVHNdata(Dataset):
    """
    A digit classification benchmark dataset that contains the street view house number (SVHN) images
    
    This dataset includes 99,289 32×32 RGB images of printed digits cropped from pictures of house number plates.
    """

    def __init__(self):
        x_train, y_train = load_data("train_32x32.mat")
        x_val, y_val = load_data("test_32x32.mat")

        x_train, y_train = x_train.transpose((3,0,1,2)), y_train[:,0]
        x_val, y_val = x_val.transpose((3,0,1,2)), y_val[:,0]

        y_train[y_train == 10] = 0
        y_val[y_val == 10] = 0
        
        # Fit the OneHotEncoder
        enc = OneHotEncoder().fit(y_train.reshape(-1, 1))
        
        # Transform the label values to a one-hot-encoding scheme
        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        y_val = enc.transform(y_val.reshape(-1, 1)).toarray()
        
        x_train = tf.image.resize(x_train, [28, 28])
        x_val = tf.image.resize(x_val, [28, 28])
        
        print("SVHN : Training Set", x_train.shape)
        print("SVHN : Test Set", x_val.shape)
        
        # Calculate the total number of images
        num_images = x_train.shape[0] + x_val.shape[0]
        print("SVHN : Total Number of Images", num_images)
        
        self.x = normalize(np.vstack([x_train, x_val]))
        self.y = np.vstack([y_train, y_val])