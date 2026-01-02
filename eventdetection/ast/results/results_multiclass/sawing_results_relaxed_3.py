import numpy as np

# FOLD 1:
# Final results:
# FN: 0.0
# FP: 0.0
# TP: 4.0

prec_1 = 1.0
rec_1 = 1.0
f1_1 = 1.0

#######################################
# FOLD 2:
# Final results:
# FN: 2.0
# FP: 0.0
# TP: 1.0

prec_2 = 1.0
rec_2 = 0.3333333333333333
f1_2 = 0.5

#######################################
# FOLD 3:
# Final results:
# FN: 0.0
# FP: 0.0
# TP: 3.0

prec_3 = 1.0
rec_3 = 1.0
f1_3 = 1.0

#######################################
# OVERALL:

prec = [prec_1, prec_2, prec_3]
rec = [rec_1, rec_2, rec_3]
f1 = [f1_1, f1_2, f1_3]

print("Precision: " + str(np.mean(prec)) + " (mean) - " + str(np.std(prec)) + " (std)")
print("Recall: " + str(np.mean(rec)) + " (mean) - " + str(np.std(rec)) + " (std)")
print("F1: " + str(np.mean(f1)) + " (mean) - " + str(np.std(f1)) + " (std)")


