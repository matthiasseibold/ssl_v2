import numpy as np

# FOLD 1:
# Final results:
# FN: 11.0
# FP: 4.0
# TP: 1.0

prec_1 = 0.2
rec_1 = 0.08333333333333333
f1_1 = 0.11764705882352941

#######################################
# FOLD 2:
# Final results:
# FN: 8.0
# FP: 7.0
# TP: 6.0

prec_2 = 0.46153846153846156
rec_2 = 0.42857142857142855
f1_2 = 0.4444444444444444

#######################################
# FOLD 3:
# Final results:
# FN: 1.0
# FP: 15.0
# TP: 3.0

prec_3 = 0.16666666666666666
rec_3 = 0.21428571428571427
f1_3 = 0.1875

#######################################
# OVERALL:

prec = [prec_1, prec_2, prec_3]
rec = [rec_1, rec_2, rec_3]
f1 = [f1_1, f1_2, f1_3]

print("Precision: " + str(np.mean(prec)) + " (mean) - " + str(np.std(prec)) + " (std)")
print("Recall: " + str(np.mean(rec)) + " (mean) - " + str(np.std(rec)) + " (std)")
print("F1: " + str(np.mean(f1)) + " (mean) - " + str(np.std(f1)) + " (std)")


