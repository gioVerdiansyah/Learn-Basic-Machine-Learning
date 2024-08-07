from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation, rc
rc('animation', html='html5')

# NUM_EXAMPLES = 256
# BATCH_SIZE = 8
# STEPS = 50 # actually steps
# LR = 0.1


def animate_sgd(num_examples, batch_size, steps, learning_rate,
                true_w=3.0, true_b=2.0, seed=0):
    # Define model
    class Model(object):
        def __init__(self, w_init=-1.0, b_init=-1.0):
            self.W = tf.Variable(w_init)
            self.b = tf.Variable(b_init)

        def __call__(self, x):
            return self.W * x + self.b
            
    def loss(target_y, predicted_y):
        return tf.reduce_mean(tf.square(target_y - predicted_y))

    def train(model, inputs, outputs, learning_rate):
        with tf.GradientTape() as t:
            current_loss = loss(outputs, model(inputs))
            dW, db = t.gradient(current_loss, [model.W, model.b])
            model.W.assign_sub(learning_rate * dW)
            model.b.assign_sub(learning_rate * db)
    # Data
    inputs  = tf.random.normal(shape=[num_examples], seed=seed)
    noise   = tf.random.normal(shape=[num_examples], seed=seed+1)
    outputs = inputs * true_w + true_b + noise
    ds = (tf.data.Dataset
          .from_tensor_slices((inputs, outputs))
          .shuffle(1000, seed=seed)
          .batch(batch_size)
          .repeat())
    ds = iter(ds)
    model = Model()
    # Collect the history of W-values and b-values to plot later
    Ws, bs, xs, ys, ls = [], [], [], [], []
    # Construct plot
    fig = plt.figure(dpi=100, figsize=(8, 3))

    # Regression Line
    ax1 = fig.add_subplot(131)
    ax1.set_title("Fitted Line")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(-3, 2.5)
    ax1.set_ylim(-8, 11)
    p10, = ax1.plot(inputs, outputs, 'r.', alpha=0.1) # full dataset
    p11, = ax1.plot([], [], 'C3.') # batch, color Red
    p12, = ax1.plot([], [], 'k') # fitted line, color Black

    # Loss
    ax2 = fig.add_subplot(132)
    ax2.set_title("Training Loss")
    ax2.set_xlabel("Batches Seen")
    ax2.set_xlim(0, steps)
    ax2.set_ylim(0, 40)
    p20, = ax2.plot([], [], 'C0') # color Blue

    # Weights
    ax3 = fig.add_subplot(133)
    ax3.set_title("Weights")
    ax3.set_xlabel("Batches Seen")
    ax3.set_xlim(0, steps)     # 
    ax3.set_ylim(-2, 4)
    ax3.plot(range(steps), [true_w for _ in range(steps)], 'C5--')
    ax3.plot(range(steps), [true_b for _ in range(steps)], 'C8--')
    p30, = ax3.plot([], [], 'C5') # W color Brown
    p30.set_label('W')
    p31, = ax3.plot([], [], 'C8') # b color Green
    p31.set_label('b')
    ax3.legend()

    fig.tight_layout()

    def init():
        return [p10]

    def update(epoch):
        x, y = next(ds)
        y_pred = model(x)
        current_loss = loss(y, y_pred)
          
        Ws.append(model.W.numpy())
        bs.append(model.b.numpy())
        xs.append(x.numpy())
        ys.append(y_pred.numpy())
        ls.append(current_loss.numpy())
        p11.set_data(x.numpy(), y.numpy())
        inputs = tf.linspace(-3.0, 2.5, 30)
        p12.set_data(inputs, Ws[-1]*inputs + bs[-1])
        p20.set_data(range(epoch), ls)
        p30.set_data(range(epoch), Ws)
        p31.set_data(range(epoch), bs)

        train(model, x, y, learning_rate=learning_rate)
        #   print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
        #         (epoch, Ws[-1], bs[-1], current_loss))
        
        return p11, p12, p20

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(1, steps),
        init_func=init,
        blit=True,
        interval=100,
    )
    plt.close()
    return ani

rc('animation', html='html5')


def animate_curve_fitting(model,
                             X, y,
                             batch_size=64,
                             epochs=16,
                             lr=0.005,
                             shuffle_buffer=5000,
                             seed=0,
                             verbose=1):
    num_examples = X.shape[0]
    steps_per_epoch = num_examples // batch_size
    total_steps = steps_per_epoch * epochs
    
    ds = (tf.data.Dataset
          .from_tensor_slices((X, y))
          .repeat()
          .cache()
          .shuffle(shuffle_buffer, seed=seed)
          .batch(batch_size))
    ds_iter = ds.as_numpy_iterator()

    x_min = X.min()
    x_max = X.max()
    X_pop = np.linspace(x_min, x_max, 1000)
    y_min = y.min()
    y_max = y.max()

    # Parameters
    xs = []
    ys = []
    curves = []
    # Callback to save parameters
    def save_params(batch, logs):
        x, y = next(ds_iter)
        xs.append(x.squeeze())
        ys.append(y.squeeze())
        curve = model.predict(X_pop)
        curves.append(curve)

    save_params_cb = keras.callbacks.LambdaCallback(
        on_batch_begin=save_params,
    )

    # Train model to collect parameters
    model.fit(
        ds,
        epochs=epochs,
        callbacks=[save_params_cb],
        steps_per_epoch=steps_per_epoch,
        verbose=verbose,
    )

    # Create Figure
    fig = plt.figure(dpi=150, figsize=(4, 3))
    # Regression Curve
    ax1 = fig.add_subplot(111)
    ax1.set_title("Fitted Curve")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    p10, = ax1.plot(X, y, 'r.', alpha=0.1) # full dataset
    p11, = ax1.plot([], [], 'C3.') # batch
    p12, = ax1.plot([], [], 'k') # fitted line
    # Complete Figure
    fig.tight_layout()

    def init():
        return [p10]

    def update(frame):
        x = xs[frame]
        y = ys[frame]
        p11.set_data(x, y)
        p12.set_data(X_pop, curves[frame])
        return p11, p12

    ani = \
        animation.FuncAnimation(
            fig,
            update,
            frames=range(1, total_steps),
            init_func=init,
            blit=True,
            interval=100,
        )
    plt.close()

    return ani