import numpy as np

# FOLD 1:
# Final results:
# FN: 5.0
# FP: 0.0
# TP: 86.0

prec_1 = 1.0
rec_1 = 0.945054945054945
f1_1 = 0.9717514124293786

#######################################
# FOLD 2:
# Final results:
# FN: 6.0
# FP: 3.0
# TP: 75.0

prec_2 = 0.9615384615384616
rec_2 = 0.9259259259259259
f1_2 = 0.9433962264150944

#######################################
# FOLD 3:
# Final results:
# FN: 0.0
# FP: 1.0
# TP: 95.0

prec_3 = 0.9895833333333334
rec_3 = 1.0
f1_3 = 0.9947643979057592

#######################################
# OVERALL:

prec = [prec_1, prec_2, prec_3]
rec = [rec_1, rec_2, rec_3]
f1 = [f1_1, f1_2, f1_3]

print("Precision: " + str(np.mean(prec)) + " (mean) - " + str(np.std(prec)) + " (std)")
print("Recall: " + str(np.mean(rec)) + " (mean) - " + str(np.std(rec)) + " (std)")
print("F1: " + str(np.mean(f1)) + " (mean) - " + str(np.std(f1)) + " (std)")


