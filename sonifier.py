import os
import numpy as np

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

class Sonifier:

    def __init__(
        self, 
        sample_rate = 44100,
        buffer_size = 2048,
        bit_depth = -16,
        base_freq = 110.0
    ) -> None:
        """
        Initialize the sonifier.

        Args:
            sample_rate: Audio sample rate in Hz.
            buffer_size: pygame mixer buffer size (lower = less latency, more CPU).
            base_freq: Base frequency for the drone in Hz (default A2 = 110Hz).
        """
        if not PYGAME_AVAILABLE:
            self.enabled = False
            return
        
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.base_freq = base_freq
        self.enabled = True

        # Pentatonic scale intervals (pleasant, non-dissonant)
        self.scale_ratios = [1, 9/8, 5/4, 3/2, 5/3, 2]

        # Initialize pygame mixer
        pygame.mixer.pre_init(sample_rate, -16, 2, buffer_size)
        pygame.mixer.init()

        # Dedicated channels for layered sound
        self.drone_channel = pygame.mixer.Channel(0)
        self.activity_channel = pygame.mixer.Channel(1)

    def _generate_tone(
        self,
        freq: float,
        duration: float,
        amplitude: float = 0.3,
        pan: float = 0.5,
    ) -> pygame.mixer.Sound:
        """
        Generate a sine wave tone with harmonics.

        Args:
            freq: Fundamental frequency in Hz.
            duration: Tone duration in seconds.
            amplitude: Volume (0.0 to 1.0).
            pan: Stereo position (0.0 = left, 0.5 = center, 1.0 = right).

        Returns:
            pygame.mixer.Sound object ready to play.
        """
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, dtype=np.float32)

        # Sine wave with harmonics for warmth
        wave = amplitude * (
            0.6 * np.sin(2 * np.pi * freq * t) +      # Fundamental
            0.3 * np.sin(4 * np.pi * freq * t) +      # 2nd harmonic
            0.1 * np.sin(6 * np.pi * freq * t)        # 3rd harmonic
        )

        # Fade in/out to prevent audio clicks
        fade_samples = min(100, n_samples // 10)
        if fade_samples > 0:
            wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
            wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        # Stereo panning
        left = wave * (1 - pan)
        right = wave * pan
        stereo = np.column_stack((left, right))

        # Convert to 16-bit signed integers for pygame
        stereo_int = (stereo * 32767).astype(np.int16)
        return pygame.mixer.Sound(buffer=stereo_int)


    def update(
        self,
        population: int,
        max_population: int,
        births: int, 
        deaths: int,
        interval_ms: int
    ) -> None:
        
        if not self.enabled:
            return
        duration = interval_ms / 1000.0

        ## Pitch = Population
        if max_population > 0:
            pop_density = population / max_population
        else:
            pop_density = 0
            
        scale_index = int(pop_density * (len(self.scale_ratios) - 1))
        scale_index = max(0, min(scale_index, len(self.scale_ratios) - 1))
        freq = self.base_freq * self.scale_ratios[scale_index]

        ## Volume = Activity
        if max_population > 0:
            activity = (births + deaths) / max_population
        else:
            activity = 0
        amplitude = 0.1 + (0.4 * activity)

        ## Stereo Pan = Volatility
        if population > 0:
            volatility = abs(births - deaths) / population
        else: 
            volatility = 0
        pan = 0.5 + 0.3 * min(volatility, 1.0) * (1 if births > deaths else -1)
        pan = max(0.1, min(0.9, pan))

        # Generate and play drone
        tone = self._generate_tone(freq, duration, amplitude, pan)
        self.drone_channel.play(tone)

        # --- Activity layer: high octave on significant births ---
        if births > 10:
            birth_amp = 0.1 * min(births / 100, 1.0)
            birth_tone = self._generate_tone(
                freq * 2,  # One octave up
                duration * 0.3,
                birth_amp,
                0.5,
            )
            self.activity_channel.play(birth_tone)
    
    def toggle(self) -> bool:
        """
        Toggle sound on/off.

        Returns:
            New enabled state.
        """
        if not PYGAME_AVAILABLE:
            return False

        self.enabled = not self.enabled
        if not self.enabled:
            self.drone_channel.stop()
            self.activity_channel.stop()
        return self.enabled

    def cleanup(self) -> None:
        """Clean up pygame mixer resources."""
        if PYGAME_AVAILABLE and pygame.mixer.get_init():
            pygame.mixer.quit()



