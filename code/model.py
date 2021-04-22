from utils import *
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, losses, metrics, models, optimizers

class Model:
    
    def __init__(self, classes=10):
        self.cce = losses.CategoricalCrossentropy(from_logits=True)
        self.acc = metrics.CategoricalAccuracy()
        
    def sampling(self, ds, expected_size, buffer_size=2048):
        ds_size = ds.cardinality().numpy()
        
        if ds_size < expected_size:
            ds = ds.repeat(expected_size//ds_size + 1)
            
        ds = ds.shuffle(buffer_size=buffer_size).take(expected_size)
        
        return ds
        
    def create_ds(self, x, y, batch_size=200, train_size=20000, val_size=2000):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, shuffle=True)
        
        # Prepare the training dataset.
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_ds = self.sampling(train_ds, train_size).batch(batch_size)

        # Prepare the validation dataset.
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_ds = self.sampling(val_ds, val_size).batch(batch_size)
        
        return train_ds, val_ds
    
    def loss_sol(self, y_true, y_hat):
        loss_true = self.cce(y_true, y_hat)
        
        return loss_true
    
    def build_solver(self, img_height=28, img_width=28, classes=10, name="Naive NN"):
        self.sol = models.Sequential([
            layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=(img_height, img_width, 3)),
            layers.LeakyReLU(alpha=0.3),
            layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.3),
            layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.3),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(classes)
        ], name=name)
        
        self.opt_sol = optimizers.Adam(learning_rate=0.001)
        
    def step_solver(self, x, y, trainable=True):
        
        with tf.GradientTape() as t_sol:

            y_hat = self.sol(x, training=trainable)
            
            ls = self.loss_sol(y, y_hat)
            self.acc.reset_states()
            self.acc.update_state(y, y_hat)
            acc = self.acc.result().numpy()
            
            if trainable :
                grad_sol = t_sol.gradient(ls, self.sol.trainable_variables)
                self.opt_sol.apply_gradients(zip(grad_sol, self.sol.trainable_variables))
            
            return ls, acc
        
    def train_solver(self, data, ckpt_name, prev_scholars=None, epochs=5, verbose=1):
        global parampath
        
        if not hasattr(self, 'task'):
            print("┕Prepare data input pipeline...")
            self.prepare_batch(data, prev_scholars)
        
        task_name, train_ds, val_ds = self.task
        
        if verbose == 2 :
            self.sol.summary()
            print()
            
        train_steps = train_ds.cardinality().numpy()
        val_steps = val_ds.cardinality().numpy()
        
        if prev_scholars != None :
            for sch in prev_scholars:
                prev_task, _, _ = sch.task
                self.history['loss_'+prev_task] = []
                self.history['accuracy_'+prev_task] = []
                
        self.history['loss_'+task_name] = []
        self.history['accuracy_'+task_name] = []
            
        for epoch in range(epochs):
            
            # Iterate over the batches of the dataset.
            for step, (batch_x, batch_y) in enumerate(train_ds):
                ls, acc = self.step_solver(batch_x, batch_y)
                
                msg_prefix = '▶%s Ep.%2d'%(self.sol.name, epoch + 1)
                msg_suffix = 'Train [loss :%.3f accuracy:%.3f]'%(ls, acc)
                
                self.print_progress(msg_prefix, msg_suffix, step, train_steps)
                
                if (step+1)%5 == 0 :
                    val_loss = 0.
                    val_acc = 0.
            
                    # Iterate over the batches of the dataset.
                    for _, (batch_vx, batch_vy) in enumerate(val_ds):
                        vls, vacc = self.step_solver(batch_vx, batch_vy, trainable=False)
                        val_loss += vls.numpy()/val_steps
                        val_acc += vacc/val_steps
                
                    self.history['loss_'+task_name].append(val_loss)
                    self.history['accuracy_'+task_name].append(val_acc)
            
                    if step + 1 == train_steps :
                        print('  Test [loss:%.3f accuracy:%.3f]'%(val_loss, val_acc), end='')
            
                    if prev_scholars != None :
                    
                        for sch in prev_scholars:
                            prev_task, _, prev_val_ds = sch.task
                            prev_val_loss = 0.
                            prev_val_acc = 0.
                
                            prev_val_steps = prev_val_ds.cardinality().numpy()
            
                            # Iterate over the batches of the dataset.
                            for _, (batch_pvx, batch_pvy) in enumerate(prev_val_ds):
                                pvls, pvacc = self.step_solver(batch_pvx, batch_pvy, trainable=False)
                                prev_val_loss += pvls.numpy()/prev_val_steps
                                prev_val_acc += pvacc/prev_val_steps
                        
                            self.history['loss_'+prev_task].append(prev_val_loss)
                            self.history['accuracy_'+prev_task].append(prev_val_acc)
                    
                            if step + 1 == train_steps :
                                print('  %s [loss:%.3f accuracy:%.3f]'%(prev_task, prev_val_loss, prev_val_acc), end='')
                    
            print()
            
        ckpt = os.path.join(parampath, ckpt_name, 'sol_'+task_name+'.h5')
        self.sol.save(ckpt)
        
        return self.history
    
    def print_progress(self, msg_prefix, msg_suffix, step, tot_steps, bar_len=36):
        filled_len = int(round(bar_len * (step+1) / float(tot_steps)))
        bar = '#' * filled_len + '-' * (bar_len - filled_len)
        print('\r%s [%s] %s'%(msg_prefix, bar, msg_suffix), end='')
    
        
class NaiveNN(Model):
    """
    A typical neural network architecture
    """
    
    def __init__(self, classes=10):
        self.build_solver(classes=classes)
        super().__init__()
        
    def prepare_batch(self, data, prev_scholars=None, batch_size=200):
        task_name, x, y = data.getTask()
        self.history = {}
        
        train_ds, val_ds = self.create_ds(x, y, batch_size=batch_size)
        self.task = (task_name, train_ds, val_ds)
    

class Scholar(Model):
    """
    A Generator-Solver pair that produces fake data and desired target pairs.
    """

    def __init__(self, classes=10):
        self.build_generator(classes=classes)
        self.build_solver(classes=classes, name="Solver")
        self.ce = losses.BinaryCrossentropy(from_logits=True)
        self.gen_acc = metrics.BinaryAccuracy()
        super().__init__()
        
    def copy(self, prev_scholar):
        self.sol = models.clone_model(prev_scholar.sol)
        self.sol.set_weights(prev_scholar.sol.get_weights())
        self.gen = models.clone_model(prev_scholar.gen)
        self.gen.set_weights(prev_scholar.gen.get_weights())
        self.disc = models.clone_model(prev_scholar.disc)
        self.disc.set_weights(prev_scholar.disc.get_weights())
        
    def loss_disc(self, y_hat_real, y_hat_fake):
        loss_real = self.ce(tf.ones_like(y_hat_real), y_hat_real)
        loss_fake = self.ce(tf.zeros_like(y_hat_fake), y_hat_fake)
        
        return loss_real + loss_fake
    
    def loss_gen(self, y_hat_fake):
        loss_fake = self.ce(tf.ones_like(y_hat_fake), y_hat_fake)
        
        return loss_fake
        
    def step_generator(self, x_real, noise_size=100):
        batch_size=tf.shape(x_real).numpy()[0]
        noise = tf.random.normal([batch_size, noise_size])
        
        with tf.GradientTape() as t_gen, tf.GradientTape() as t_disc:
            x_fake = self.gen(noise, training=True)

            y_hat_real = self.disc(x_real, training=True)
            y_hat_fake = self.disc(x_fake, training=True)
            
            y_hat = tf.stack([y_hat_real, y_hat_fake])
            y = tf.stack([tf.zeros_like(y_hat_real), tf.ones_like(y_hat_fake)])

            ld, lg = self.loss_disc(y_hat_real, y_hat_fake), self.loss_gen(y_hat_fake)
            
            self.gen_acc.reset_states()
            self.gen_acc.update_state(y, y_hat)
            gacc = self.gen_acc.result().numpy()
            
            grad_gen = t_gen.gradient(lg, self.gen.trainable_variables)
            grad_disc = t_disc.gradient(ld, self.disc.trainable_variables)

            self.opt_gen.apply_gradients(zip(grad_gen, self.gen.trainable_variables))
            self.opt_disc.apply_gradients(zip(grad_disc, self.disc.trainable_variables))
            
            return ld, lg, gacc
        
    def build_generator(self, img_height=28, img_width=28, noise_size=100, classes=10):
        self.gen = models.Sequential([
            layers.Dense(7*7*64, use_bias=False, input_shape=(noise_size,)),
            layers.LeakyReLU(alpha=0.3),
            layers.Reshape((7, 7, 64)),
            layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False),
            layers.LeakyReLU(alpha=0.3),
            layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.LeakyReLU(alpha=0.3),
            layers.Conv2DTranspose(filters=3, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        ], name="Generator")
        
        assert self.gen.output_shape == (None, img_height, img_width, 3)
        
        self.opt_gen = optimizers.Adam(learning_rate=0.0001)
        
        self.disc = models.Sequential([
            layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(img_height, img_width, 3)),
            layers.LeakyReLU(alpha=0.3),
            layers.Dropout(0.3),
            layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.3),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1)
        ], name="Discriminator")
        
        self.opt_disc = optimizers.Adam(learning_rate=0.0001)
        
    def prepare_batch(self, data, prev_scholars=None, batch_size=200, noise_size=100, prop=0.9):
        task_name, x, y = data.getTask()
        self.history = {}
        
        if prev_scholars != None :
            self.copy(prev_scholars[-1])
            
            true_samples = y.shape[0]
            fake_samples = int((1-prop)/prop*true_samples)
            
            for i in range(0, fake_samples, batch_size):
                noise = tf.random.normal([min(batch_size, fake_samples-i), noise_size])
                x_fake = self.gen(noise, training=False)
                y_fake = self.sol(x_fake, training=False).numpy()
                y_fake = (y_fake == y_fake.max(axis=1)[:,None]).astype(int)
                
                x = np.vstack([x, x_fake])
                y = np.vstack([y, y_fake])
                
        train_ds, val_ds = self.create_ds(x, y, batch_size=batch_size)
        
        self.task = (task_name, train_ds, val_ds)
        
    def train_generator(self, data, ckpt_name, epochs=50, verbose=1):
        global parampath
        
        if not hasattr(self, 'task'):
            self.prepare_batch(data)
        
        task_name, train_ds, val_ds = self.task
        
        if verbose == 2 :
            self.gen.summary()
            print()
            self.disc.summary()
            print()
            
        train_steps = train_ds.cardinality().numpy()
            
        for epoch in range(epochs):
            
            # Iterate over the batches of the dataset.
            for step, (batch_x, batch_y) in enumerate(train_ds):
                lg, ld, gacc = self.step_generator(batch_x)
                
                msg_prefix = '▶Generator  Ep.%2d'%(epoch + 1)
                msg_suffix = 'Generator [loss:%.3f]  Discriminator [loss:%.3f accuracy:%.3f]'%(lg, ld, gacc)
                
                self.print_progress(msg_prefix, msg_suffix, step, train_steps)
                
            print()
            
        for nn, name in zip([self.gen, self.disc], ['gen', 'disc']):
            ckpt = os.path.join(parampath, ckpt_name, name+'_'+task_name+'.h5')
            nn.save(ckpt)
    
    def generate_images(self, nrows=2, ncols=8, num_samples = 100, noise_size=100):
        noise = tf.random.normal([num_samples, noise_size])
        samples = self.gen(noise, training=False).numpy()
        
        fig, axes = plt.subplots(nrows, ncols)
        
        for i, ax in enumerate(axes.flat): 
            img = recover(samples[i,:])
            ax.imshow(img)
            ax.axis('off')
        
        plt.show()