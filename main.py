import numpy as np
import tensorflow as tf
import sionna as sn
import matplotlib.pyplot as plt

# Print versions for verification
print(f"Sionna version: {sn.__version__}")
print(f"TensorFlow version: {tf.__version__}")

# Verify GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {len(gpus)}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("Warning: No GPU detected. Running on CPU.")

# System parameters
NUM_USERS = 10
NUM_ANTENNAS_BS = 64
NUM_IRS_ELEMENTS = 100
FREQUENCY = 0.3e12  # 0.3 THz
BANDWIDTH = 20e9    # 20 GHz
TX_POWER_dBm = 20
NOISE_POWER_dBm = -90
ROOM_SIZE = 20
NUM_TIME_FRAMES = 100
C = 3e8  # Speed of light (m/s)

# Convert dBm to Watts
TX_POWER = 10 ** ((TX_POWER_dBm - 30) / 10)
NOISE_POWER = 10 ** ((NOISE_POWER_dBm - 30) / 10)

# Define neural network model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(NUM_ANTENNAS_BS * 2,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(NUM_ANTENNAS_BS * 2 + NUM_IRS_ELEMENTS * 2)
    ])
    return model

# Generate CSI with Sionna Ray Tracing
def generate_csi_sionna(time_frame):
    np.random.seed(time_frame)
    scene = sn.rt.load_scene("__empty__", dtype=tf.complex64)
    scene.frequency = float(FREQUENCY)

    scene.tx_array = sn.rt.PlanarArray(num_rows=8, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5,
                                       pattern="tr38901", polarization="V")
    scene.rx_array = sn.rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
                                       pattern="dipole", polarization="V")
    irs_array = sn.rt.PlanarArray(num_rows=10, num_cols=10, vertical_spacing=0.5, horizontal_spacing=0.5,
                                  pattern="dipole", polarization="V")

    bs_pos = np.array([ROOM_SIZE / 2, ROOM_SIZE / 2, 2.0])
    scene.add(sn.rt.Transmitter(name="bs", position=bs_pos, orientation=[0, 0, 0]))

    user_pos = np.random.uniform(0, ROOM_SIZE, (NUM_USERS, 3))
    user_pos[:, 2] = 1.0
    for k in range(NUM_USERS):
        scene.add(sn.rt.Receiver(name=f"ue-{k}", position=user_pos[k], orientation=[0, 0, 0]))

    irs_pos = np.array([[0, ROOM_SIZE / 2, 2.0], [ROOM_SIZE, ROOM_SIZE / 2, 2.0]])
    irs_list = []
    for i in range(2):
        ris = sn.rt.RIS(name=f"ris-{i}", position=irs_pos[i], orientation=[0, 0, 0], num_rows=10, num_cols=10)
        scene.add(ris)
        irs_list.append(ris)

    for i in range(2):
        scene.add(sn.rt.Receiver(name=f"irs-rx-{i}", position=irs_pos[i], orientation=[0, 0, 0]))

    paths = scene.compute_paths(max_depth=2, method="fibonacci", num_samples=10e6, los=True, reflection=True, ris=True)

    a_direct, _ = paths.cir(los=True, reflection=True, diffraction=True, scattering=True, ris=False)
    print(f"a_direct shape: {a_direct.shape}, total elements: {tf.size(a_direct)}")
    a_direct_ue = a_direct[:, :NUM_USERS, :, :, :, :, :]
    print(f"a_direct_ue shape: {a_direct_ue.shape}, total elements: {tf.size(a_direct_ue)}")
    h_direct = tf.reshape(tf.squeeze(a_direct_ue), [NUM_USERS, NUM_ANTENNAS_BS])  # [10, 64]

    a_ris, _ = paths.cir(los=False, reflection=False, diffraction=False, scattering=False, ris=True, cluster_ris_paths=True)
    print(f"a_ris shape: {a_ris.shape}, total elements: {tf.size(a_ris)}")
    h_irs = tf.zeros([NUM_USERS, 2, NUM_IRS_ELEMENTS], dtype=tf.complex64)  # [10, 2, 100]

    H_bs_irs = np.zeros((2, NUM_IRS_ELEMENTS, NUM_ANTENNAS_BS), dtype=complex)
    a_irs_rx = a_ris[:, NUM_USERS:, :, :, :, :, :]
    print(f"a_irs_rx shape: {a_irs_rx.shape}, total elements: {tf.size(a_irs_rx)}")
    for i in range(2):
        a_bs_irs = a_irs_rx[:, i, :, :, :, :, :]
        print(f"a_bs_irs shape for IRS {i}: {a_bs_irs.shape}, total elements: {tf.size(a_bs_irs)}")
        H_bs_irs[i] = tf.zeros([NUM_IRS_ELEMENTS, NUM_ANTENNAS_BS], dtype=tf.complex64)

    return h_direct.numpy(), h_irs.numpy(), H_bs_irs

# Compute sum-rate (TensorFlow)
@tf.function
def compute_sum_rate_tf(h_direct, h_irs, H_bs_irs, w, phi):
    rates = []
    for k in range(NUM_USERS):
        w_k = tf.expand_dims(w[k], axis=-1)  # [64, 1]
        h_eff_irs = tf.zeros([NUM_ANTENNAS_BS], dtype=tf.complex64)  # [64]
        for i in range(2):
            h_irs_k_i = tf.expand_dims(h_irs[k, i], axis=0)  # [1, 100]
            phi_diag = tf.linalg.diag(phi[i])  # [100, 100]
            contrib = h_irs_k_i @ phi_diag @ H_bs_irs[i] @ w_k  # [1, 64]
            h_eff_irs += tf.squeeze(contrib)  # [64]
        h_eff = h_direct[k] + h_eff_irs  # [64]
        signal = tf.abs(tf.reduce_sum(tf.math.conj(h_eff) * w_k)) ** 2
        interference = tf.reduce_sum([
            tf.abs(tf.reduce_sum(tf.math.conj(h_eff) * tf.expand_dims(w[j], axis=-1))) ** 2 
            for j in range(NUM_USERS) if j != k])
        snr = (signal * TX_POWER) / (interference * TX_POWER + NOISE_POWER)
        rate = BANDWIDTH * tf.math.log(1 + snr) / tf.math.log(2.0)
        rates.append(rate)
    return tf.reduce_sum(rates) / 1e9

# Custom loss function
def sum_rate_loss(y_true, y_pred, h_direct, h_irs, H_bs_irs):
    w_real = y_pred[:, :NUM_ANTENNAS_BS]
    w_imag = y_pred[:, NUM_ANTENNAS_BS:NUM_ANTENNAS_BS*2]
    phi_angles = y_pred[:, NUM_ANTENNAS_BS*2:]
    
    w = tf.complex(w_real, w_imag)  # [10, 64]
    phi = [tf.exp(tf.complex(0.0, tf.reduce_mean(phi_angles[:, i*NUM_IRS_ELEMENTS:(i+1)*NUM_IRS_ELEMENTS], axis=0))) 
           for i in range(2)]  # [100] per IRS
    
    sum_rate = compute_sum_rate_tf(h_direct, h_irs, H_bs_irs, w, phi)
    return -sum_rate

# WMMSE-like initialization
def wmmse_initialization(h_direct, h_irs, H_bs_irs, tx_power, noise_power, num_iters=5):
    num_users = h_direct.shape[0]
    w = [h_direct[k] / np.linalg.norm(h_direct[k]) for k in range(num_users)]
    phi = [np.ones(NUM_IRS_ELEMENTS, dtype=complex) for _ in range(2)]

    for _ in range(num_iters):
        for k in range(num_users):
            h_eff = h_direct[k] + np.sum([h_irs[k, i] @ np.diag(phi[i]) @ H_bs_irs[i] for i in range(2)], axis=0)
            interference = sum([np.abs(np.vdot(h_eff, w[j])) ** 2 for j in range(num_users) if j != k])
            sinr = (np.abs(np.vdot(h_eff, w[k])) ** 2 * tx_power) / (interference * tx_power + noise_power)
            w[k] = h_eff / np.linalg.norm(h_eff) * np.sqrt(tx_power / num_users) / (1 + 1/sinr)
            w[k] /= np.linalg.norm(w[k])

        for i in range(2):
            phi_sum = np.zeros(NUM_IRS_ELEMENTS, dtype=complex)
            for k in range(num_users):
                phi_sum += h_irs[k, i].conj() @ H_bs_irs[i] @ w[k]
            phi[i] = np.exp(1j * np.angle(phi_sum))

    return w, phi

# Compute sum-rate (NumPy)
def compute_sum_rate(h_direct, h_irs, H_bs_irs, w, phi):
    num_users = h_direct.shape[0]
    rates = []
    for k in range(num_users):
        h_eff = h_direct[k] + np.sum([h_irs[k, i] @ np.diag(phi[i]) @ H_bs_irs[i] for i in range(2)], axis=0)
        signal = np.abs(np.vdot(h_eff, w[k])) ** 2
        interference = sum([np.abs(np.vdot(h_eff, w[j])) ** 2 for j in range(num_users) if j != k])
        snr = (signal * TX_POWER) / (interference * TX_POWER + NOISE_POWER)
        rate = BANDWIDTH * np.log2(1 + snr)
        rates.append(rate)
    return np.sum(rates) / 1e9

# Federated Learning algorithm
def federated_learning():
    with tf.device('/GPU:0'):
        model = create_model()
        
        sum_rates_fl = []
        sum_rates_baseline = []

        for t in range(NUM_TIME_FRAMES):
            h_direct_np, h_irs_np, H_bs_irs_np = generate_csi_sionna(t)
            h_direct = tf.constant(h_direct_np, dtype=tf.complex64)
            h_irs = tf.constant(h_irs_np, dtype=tf.complex64)
            H_bs_irs = tf.constant(H_bs_irs_np, dtype=tf.complex64)

            w_target, phi_target = wmmse_initialization(h_direct_np, h_irs_np, H_bs_irs_np, TX_POWER, NOISE_POWER)
            phi_target_angles = np.concatenate([np.angle(phi_target[0]), np.angle(phi_target[1])])
            y_true = np.array([np.concatenate([w_target[k].real, w_target[k].imag, phi_target_angles]) 
                              for k in range(NUM_USERS)])

            csi_inputs = np.array([np.concatenate([h_direct_np[k].real, h_direct_np[k].imag]) 
                                 for k in range(NUM_USERS)])

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            for _ in range(20):
                with tf.GradientTape() as tape:
                    y_pred = model(csi_inputs, training=True)
                    loss = sum_rate_loss(y_true, y_pred, h_direct, h_irs, H_bs_irs)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            y_pred = model.predict(csi_inputs, verbose=0)
            w = []
            phi = [np.zeros(NUM_IRS_ELEMENTS, dtype=complex) for _ in range(2)]
            for k in range(NUM_USERS):
                w_real = y_pred[k, :NUM_ANTENNAS_BS]
                w_imag = y_pred[k, NUM_ANTENNAS_BS:NUM_ANTENNAS_BS*2]
                w_k = w_real + 1j * w_imag
                w.append(w_k / np.linalg.norm(w_k))
            
            phi_angles = np.mean(y_pred[:, NUM_ANTENNAS_BS*2:], axis=0)
            phi[0] = np.exp(1j * phi_angles[:NUM_IRS_ELEMENTS])
            phi[1] = np.exp(1j * phi_angles[NUM_IRS_ELEMENTS:])

            sum_rate_fl = compute_sum_rate(h_direct_np, h_irs_np, H_bs_irs_np, w, phi)
            sum_rates_fl.append(sum_rate_fl)

            w_baseline = [h_direct_np[k] / np.linalg.norm(h_direct_np[k]) for k in range(NUM_USERS)]
            phi_baseline = [np.ones(NUM_IRS_ELEMENTS, dtype=complex) for _ in range(2)]
            sum_rate_baseline = compute_sum_rate(h_direct_np, h_irs_np, H_bs_irs_np, w_baseline, phi_baseline)
            sum_rates_baseline.append(sum_rate_baseline)

            print(f"Time Frame {t+1}/{NUM_TIME_FRAMES}: FL Sum-rate = {sum_rate_fl:.2f} Gbps, Baseline = {sum_rate_baseline:.2f} Gbps")

        plt.figure(figsize=(10, 6))
        plt.plot(range(NUM_TIME_FRAMES), sum_rates_fl, label='FL-Driven THz-IRS', color='blue')
        plt.plot(range(NUM_TIME_FRAMES), sum_rates_baseline, label='Baseline (MRT)', color='red')
        plt.xlabel('Time Frame')
        plt.ylabel('Sum-rate (Gbps)')
        plt.title('Sum-rate Comparison with Sionna RT')
        plt.legend()
        plt.grid(True)
        plt.savefig('sum_rate_plot.png')
        print("Plot saved as 'sum_rate_plot.png'")
        plt.close()

        return sum_rates_fl, sum_rates_baseline

# Run simulation
if __name__ == "__main__":
    sum_rates_fl, sum_rates_baseline = federated_learning()
