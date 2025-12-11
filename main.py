from gpe_fpe_analysis import *
from measure_time import *
from wind_example import *


############# Computation time Fig 2, 3 and 4
pitch_order_comp_time(methods = ['Fast-Exact', 'Fast-Approx1', 'Naive'])
plot_pitch_order_comp_time(methods = ['Fast-Exact', 'Fast-Approx1', 'Naive'])
Ar_order_comp_time(methods = ['Fast-Exact', 'Fast-Approx1', 'Naive'])
plot_Ar_order_comp_time(methods = ['Fast-Exact', 'Fast-Approx1', 'Naive'])
segment_length_comp_time(methods = ['Fast-Exact', 'Fast-Approx1', 'Naive'])
plot_segment_length_comp_time(methods = ['Fast-Exact', 'Fast-Approx1', 'Naive'])

############# Fig 5 and 6
gpe_fpe_vs_snr(methods = ['Fast-Exact', 'Fast-Approx1'])
plot_gpe_fpe_vs_snr(methods = ['Fast-Exact', 'Fast-Approx1'])
gpe_vs_f0_min(methods = ['Fast-Exact', 'Fast-Approx1'])
plot_gpe_vs_f0_min(methods = ['Fast-Exact', 'Fast-Approx1'])

############# Fig 7 - Application example: speech in wind noise
# create_noisy_speech_files()
speech_wind_example()
plot_speech_wind_example()


