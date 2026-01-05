import numpy as np

# FOLD 1:
# Final results:
# FN: 26.0
# FP: 5.0
# TP: 65.0

prec_1 = 0.9285714285714286
rec_1 = 0.7142857142857143
f1_1 = 0.8074534161490683


#######################################
# FOLD 2:
# Final results:
# FN: 32.0
# FP: 10.0
# TP: 49.0

prec_2 = 0.8305084745762712
rec_2 = 0.6049382716049383
f1_2 = 0.7

#######################################
# FOLD 3:
# Final results:
# FN: 3.0
# FP: 3.0
# TP: 92.0

prec_3 = 0.968421052631579
rec_3 = 0.968421052631579
f1_3 = 0.968421052631579

#######################################
# OVERALL:

prec = [prec_1, prec_2, prec_3]
rec = [rec_1, rec_2, rec_3]
f1 = [f1_1, f1_2, f1_3]

print("Precision: " + str(np.mean(prec)) + " (mean) - " + str(np.std(prec)) + " (std)")
print("Recall: " + str(np.mean(rec)) + " (mean) - " + str(np.std(rec)) + " (std)")
print("F1: " + str(np.mean(f1)) + " (mean) - " + str(np.std(f1)) + " (std)")


