from scipy.optimize import minimize
from code_2.constrain_q3 import *
from code_2.objective_problem_q3 import objective
import pandas as pd

x = np.zeros(27)
b = (0.0, None)
bounds = (b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b)

con1 = {'type': 'ineq', 'fun': constrain_1}
con2 = {'type': 'ineq', 'fun': constrain_2}
con3 = {'type': 'ineq', 'fun': constrain_3}
con4 = {'type': 'ineq', 'fun': constrain_4}
con5 = {'type': 'ineq', 'fun': constrain_5}
con6 = {'type': 'ineq', 'fun': constrain_6}
con7 = {'type': 'ineq', 'fun': constrain_7}
con8 = {'type': 'ineq', 'fun': constrain_8}
con9 = {'type': 'eq', 'fun': constrain_A}
con10 = {'type': 'eq', 'fun': constrain_B}
con11 = {'type': 'eq', 'fun': constrain_C}
con20 = {'type': 'eq', 'fun': constrain_add}

con12 = {'type': 'ineq', 'fun': constrain2_1}
con13 = {'type': 'ineq', 'fun': constrain2_2}
con14 = {'type': 'ineq', 'fun': constrain2_3}
con15 = {'type': 'ineq', 'fun': constrain2_4}
con16 = {'type': 'ineq', 'fun': constrain2_5}
con17 = {'type': 'ineq', 'fun': constrain2_6}
con18 = {'type': 'ineq', 'fun': constrain2_7}
con19 = {'type': 'ineq', 'fun': constrain2_8}
cons = ([
    con1, con2, con3, con4, con5, con6, con7, con8, con9, con10, con11,  # 第一问
    con12, con13, con14, con15, con16, con17, con18, con19,
    con20
])
solution = minimize(objective, x, method='SLSQP', bounds=bounds, constraints=cons, options={'disp': True}, tol=1E-30)
result = solution.x
result1 = result[:24]
result2 = result[24:]
result1 = result1.reshape(3, 8)
col = pd.read_csv('distance_result.csv').columns[1:]
df_result = pd.DataFrame(result1, columns=col)
df_result['start'] = ['A', 'B', 'C']
df_result['add'] = result2
col = ['start','add'] + list(col)
df_result = df_result[col]
df_result.to_csv('q3.csv', float_format='%.1f', index=False)
