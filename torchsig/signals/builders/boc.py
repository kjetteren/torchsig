"""Binary Offset Carrier (BOC) Signal Builder and Modulator
"""

# TorchSig
from torchsig.signals.builder import SignalBuilder
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.utils.dsp import (
    torchsig_complex_data_type,
    multistage_polyphase_resampler,
    slice_head_tail_to_length,
    pad_head_tail_to_length,
)
from torchsig.signals.signal_lists import TorchSigSignalLists

# Third Party
import numpy as np


def bocmod_random(m: int, n: int, num_bits: int, fundamental_freq: float = 1.023e6, halfcyclesps: int = 2, phasing: str = "sin", rng = np.random.default_rng()) -> np.ndarray:
    """Generates a Binary Offset Carrier (BOC) modulated signal
    
    Args:
        m (int): Subcarrier frequency multiple (m * fundamental_freq).
        n (int): PRN code rate multiple (n * fundamental_freq).
        num_bits (int): Number of data bits to modulate.
        fundamental_freq (float, optional): Base frequency for code/subcarrier.
        halfcyclesps (int, optional): Half-cycles per symbol.
        phasing (str, optional): Subcarrier phasing ('sin' or 'cos').
        rng (optional): Seedable random number generator for reproducibility.
    
    Raises:
        ValueError: If 2*m/n is not integer
        ValueError: For invalid phasing parameter
    
    Returns:
        np.ndarray: Complex baseband BOC signal
    """
    
    # Compute kBOC and validate it's an integer
    kBOC_float = 2 * m / n
    if abs(kBOC_float - round(kBOC_float)) < 1e-5:
        kBOC = int(round(kBOC_float))
    else:
        raise ValueError("2*m/n must be an integer.")

    # Sub-carrier frequency and sample timing
    fs = m * fundamental_freq
    Ts = 1 / (2 * fs)
    time_per_sample = Ts / halfcyclesps
    epsilon = time_per_sample / 100.0

    # Generate time vector to construct the square wave
    total_samples = kBOC * halfcyclesps * num_bits
    t = np.arange(total_samples) * time_per_sample + epsilon

    # Generate square wave using the selected phasing
    if phasing == "sin":
        square_wave = np.sign(np.sin(2 * np.pi * fs * t))
    elif phasing == "cos":
        square_wave = np.sign(np.cos(2 * np.pi * fs * t))
    else:
        raise ValueError("Invalid phasing. Must be 'sin' or 'cos'.")

    # Generate random binary data bits (0 or 1) and convert to bipolar (-1, +1)
    bipolar = rng.choice([-1, 1], size=(num_bits,))

    # Upsample bits by repeating each bit kBOC*halfcyclesps times
    upsampled = np.repeat(bipolar, kBOC * halfcyclesps)

    # Modulate: multiply the upsampled bits with the square wave
    y = upsampled * square_wave
    return y.astype(torchsig_complex_data_type)


def tmbocmod_random(m: int = 6, n: int = 1, time_multiplexing_factor: float = 4/33, num_bits: int = 100, fundamental_freq: float = 1.023e6, halfcyclesps: int = 2, phasing: str = "sin", rng = np.random.default_rng()) -> np.ndarray:
    """Generates a Time-Multiplexed Binary Offset Carrier (TMBOC) modulated signal
    
    Args:
        m (int, optional): Subcarrier frequency multiple for BOC(m,n) component
        n (int, optional): PRN code rate multiple
        time_multiplexing_factor (float, optional): Ratio of BOC(m,n) to BOC(n,n)
        num_bits (int, optional): Number of data bits to modulate
        fundamental_freq (float, optional): Base frequency for code/subcarrier
        halfcyclesps (int, optional): Half-cycles per symbol
        phasing (str, optional): Subcarrier phasing ('sin' or 'cos')
        rng (optional): Random number generator
    
    Raises:
        ValueError: If time_multiplexing_factor is not 4/33
    
    Returns:
        np.ndarray: Complex baseband TMBOC signal
    """
    
    # Initialize empty array for TMBOC signal
    tmbocmod_signal = np.array([], dtype=torchsig_complex_data_type)

    # Validate time multiplexing factor
    if(time_multiplexing_factor != 4/33):
        raise ValueError("Only 4/33 time multiplexing factor is supported")
    
    # Initialize temporal index for bit counting
    temporal_index = num_bits

    # Generate TMBOC signal
    while(temporal_index > 0):
        # Generate BOC(m,n) for 1 bit
        tmbocmod_signal = np.append(tmbocmod_signal, bocmod_random(m, n, 1, fundamental_freq, halfcyclesps, phasing, rng))
        temporal_index -= 1
        # Generate BOC(n,n) for 3 bits
        tmbocmod_signal = np.append(tmbocmod_signal, bocmod_random(n, n, 3, fundamental_freq, halfcyclesps * m, phasing, rng))
        temporal_index -= 3
        # Generate BOC(m,n) for 1 bit
        tmbocmod_signal = np.append(tmbocmod_signal, bocmod_random(m, n, 1, fundamental_freq, halfcyclesps, phasing, rng))
        temporal_index -= 1
        # Generate BOC(n,n) for 1 bit
        tmbocmod_signal = np.append(tmbocmod_signal, bocmod_random(n, n, 1, fundamental_freq, halfcyclesps * m, phasing, rng))
        temporal_index -= 1
        # Generate BOC(m,n) for 1 bit
        tmbocmod_signal = np.append(tmbocmod_signal, bocmod_random(m, n, 1, fundamental_freq, halfcyclesps, phasing, rng))
        temporal_index -= 1
        # Generate BOC(n,n) for 22 bits
        tmbocmod_signal = np.append(tmbocmod_signal, bocmod_random(n, n, 22, fundamental_freq, halfcyclesps * m, phasing, rng))
        temporal_index -= 22
        # Generate BOC(m,n) for 1 bit
        tmbocmod_signal = np.append(tmbocmod_signal, bocmod_random(m, n, 1, fundamental_freq, halfcyclesps, phasing, rng))
        temporal_index -= 1
        # Generate BOC(n,n) for 3 bits
        tmbocmod_signal = np.append(tmbocmod_signal, bocmod_random(n, n, 3, fundamental_freq, halfcyclesps * m, phasing, rng))
        temporal_index -= 3
    
    # Return TMBOC signal truncated to correct length
    return tmbocmod_signal[:num_bits * 24]


def boc_modulator ( class_name:str, bandwidth:float, sample_rate:float, num_samples:int, rng=np.random.default_rng() ) -> np.ndarray:
    """Binary offset carrier (BOC) Modulator.

    Args:
        class_name (str): Name of the signal to modulate, ex: 'boc(1,1)'.
        bandwidth (float): The desired bandwidth of the signal. Must be in the same
            units as `sample_rate` and within the bounds 0 < `bandwidth` < `sample_rate`.
        sample_rate (float): The sampling rate for the IQ signal. The sample rate can use a normalized value of 1, or it
            can use a practical sample rate such as 10 MHz. However, it must use the same units as the bandwidth parameter.
        num_samples (int): The number of IQ samples to produce.
        rng (optional): Seedable random number generator for reproducibility.
    
    Raises:
        ValueError: If the class_name is not supported.

    Returns:
        np.ndarray: BOC modulated signal at the appropriate bandwidth.
    """

    # calculate final oversampling rate
    oversampling_rate = sample_rate / bandwidth

    # modulate at a nominal oversampling rate. a resampling will be applied
    # after the baseband modulation to bring it to the appropriate bandwidth.
    # NOTE: The oversampling rate is set to 2 for the baseband modulator,
    # due to the fact that the BOC signal is generated with twice the samples.
    oversampling_rate_baseband = 2

    # calculate the resampling rate needed to convert the baseband signal into proper bandwidth
    resample_rate_ideal = oversampling_rate / oversampling_rate_baseband

    # determine how many samples baseband modulator needs to implement.
    num_samples_baseband = int(np.floor(num_samples / resample_rate_ideal))

    # Determine conversion factors based on modulation type.
    if class_name == "boc(1,1)":
        # Each bit produces 4 samples, so compute num_bits accordingly.
        num_bits = int(np.ceil(num_samples_baseband / 4))
        # Generate baseband using bocmod_random; it returns num_bits * 4 samples.
        boc_signal_baseband = bocmod_random(1, 1, num_bits, rng = rng)
    elif class_name == "boc(10,5)":
        # Each bit produces 8 samples, so compute num_bits accordingly.
        num_bits = int(np.ceil(num_samples_baseband / 8))
        # Generate baseband using bocmod_random; it returns num_bits * 8 samples.
        boc_signal_baseband = bocmod_random(10, 5, num_bits, rng = rng)
    elif class_name == "tmboc(6,1,4/33)":
        # Each bit produces 24 samples, so compute num_bits accordingly.
        num_bits = int(np.ceil(num_samples_baseband / 24))
        # Generate baseband using bocmod_random; it returns num_bits * 24 samples.
        boc_signal_baseband = tmbocmod_random(num_bits = num_bits, rng = rng)
    else:
        raise ValueError(f"Unsupported BOC variant: {class_name}")

    # apply resampling
    boc_mod_correct_bw = multistage_polyphase_resampler ( boc_signal_baseband, resample_rate_ideal )
    
    # either slice or pad the signal to the proper length
    if len(boc_mod_correct_bw) > num_samples:
        boc_mod_correct_bw = slice_head_tail_to_length ( boc_mod_correct_bw, num_samples )
    else:
        boc_mod_correct_bw = pad_head_tail_to_length ( boc_mod_correct_bw, num_samples )


    baseband_signal = boc_mod_correct_bw.astype(torchsig_complex_data_type)

    return baseband_signal
    

# Builder
class BOCSignalBuilder(SignalBuilder):
    """Implements SignalBuilder() for binary offset carrier (BOC) signal.

    Attributes:
        dataset_metadata (DatasetMetadata): Parameters describing the dataset required for signal generation. 
        supported_classes (List[str]): List of supported signal classes. Set to `TorchSigSignalLists.boc_signals`.

    """

    supported_classes = TorchSigSignalLists.boc_signals

    # NOTE: The default is boc(1,1) signal for now. 
    def __init__(self, dataset_metadata: DatasetMetadata, class_name='boc(1,1)', **kwargs):
        """Initializes BOC Signal Builder. Sets `class_name = 'boc(1,1)'`.

        Args:
            dataset_metadata (DatasetMetadata): Dataset metadata.
            class_name (str, optional): Class name.
        """
        super().__init__(dataset_metadata=dataset_metadata, class_name=class_name, **kwargs)
    
    def _update_data(self) -> None:
        """Creates the IQ samples for the BOC signal based on the signal metadata fields.
        """        
        # wideband params
        sample_rate = self.dataset_metadata.sample_rate

        # signal params
        num_iq_samples_signal = self._signal.metadata.duration_in_samples
        bandwidth = self._signal.metadata.bandwidth
        class_name = self._signal.metadata.class_name

        # BOC modulator at complex baseband
        self._signal.data = boc_modulator(
            class_name,
            bandwidth,
            sample_rate,
            num_iq_samples_signal,
            self.random_generator
        )

    def _update_metadata(self) -> None:
        """Performs a signals-specific update of signal metadata.

        This does nothing because the signal does not need any 
        fields to be updated. This `_update_metadata()` must be
        implemented but is not required to create or modify any data
        or fields for this particular signal case.
        """