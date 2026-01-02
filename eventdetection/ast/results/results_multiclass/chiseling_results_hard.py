import numpy as np

# FOLD 1:
# Final results:
# FN: 6.0
# FP: 2.0
# TP: 85.0

prec_1 = 0.9154929577464789
rec_1 = 0.7142857142857143
f1_1 = 0.8024691358024691

#######################################
# FOLD 2:
# Final results:
# FN: 26.0
# FP: 9.0
# TP: 55.0

prec_2 = 0.859375
rec_2 = 0.6790123456790124
f1_2 = 0.7586206896551724

#######################################
# FOLD 3:
# Final results:
# FN: 0.0
# FP: 1.0
# TP: 95.0

prec_3 = 0.96875
rec_3 = 0.9789473684210527
f1_3 = 0.9738219895287958

#######################################
# OVERALL:

prec = [prec_1, prec_2, prec_3]
rec = [rec_1, rec_2, rec_3]
f1 = [f1_1, f1_2, f1_3]

print("Precision: " + str(np.mean(prec)) + " (mean) - " + str(np.std(prec)) + " (std)")
print("Recall: " + str(np.mean(rec)) + " (mean) - " + str(np.std(rec)) + " (std)")
print("F1: " + str(np.mean(f1)) + " (mean) - " + str(np.std(f1)) + " (std)")


