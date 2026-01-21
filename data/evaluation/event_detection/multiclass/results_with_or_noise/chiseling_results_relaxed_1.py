import numpy as np

# FOLD 1:
# Final results:
# FN: 4.0
# FP: 0.0
# TP: 87.0

prec_1 = 1.0
rec_1 = 0.9560439560439561
f1_1 = 0.9775280898876404

#######################################
# FOLD 2:
# Final results:
# FN: 11.0
# FP: 5.0
# TP: 70.0

prec_2 = 0.9333333333333333
rec_2 = 0.8641975308641975
f1_2 = 0.8974358974358975

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


