import numpy as np

# FOLD 1:
# Final results:
# FN: 7.0
# FP: 1.0
# TP: 5.0

prec_1 = 0.8333333333333334
rec_1 = 0.4166666666666667
f1_1 = 0.5555555555555556

#######################################
# FOLD 2:
# Final results:
# FN: 3.0
# FP: 5.0
# TP: 11.0

prec_2 = 0.6875
rec_2 = 0.7857142857142857
f1_2 = 0.7333333333333333

#######################################
# FOLD 3:
# Final results:
# FN: 3.0
# FP: 7.0
# TP: 11.0

prec_3 = 0.6111111111111112
rec_3 = 0.7857142857142857
f1_3 = 0.6875

#######################################
# OVERALL:

prec = [prec_1, prec_2, prec_3]
rec = [rec_1, rec_2, rec_3]
f1 = [f1_1, f1_2, f1_3]

print("Precision: " + str(np.mean(prec)) + " (mean) - " + str(np.std(prec)) + " (std)")
print("Recall: " + str(np.mean(rec)) + " (mean) - " + str(np.std(rec)) + " (std)")
print("F1: " + str(np.mean(f1)) + " (mean) - " + str(np.std(f1)) + " (std)")


