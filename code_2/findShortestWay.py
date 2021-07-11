# 寻求最短路径
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


def getCellPath(graph, s, ends):
    result = []
    for e in ends:
        l = nx.dijkstra_path(graph, source=s, target=e)
        for i in range(len(l) - 1):
            result += [(l[i], l[i + 1])]
    result = list(set(result))
    return result


distanceInfo = pd.read_csv('data/距离信息.csv')
distanceInfo.set_index('Unnamed: 0', inplace=True)
distanceInfo = distanceInfo.stack()
distanceInfo = distanceInfo.reset_index()
distanceInfo.columns = ['loc1', 'loc2', 'len']
distanceInfo['loc2'] = pd.to_numeric(distanceInfo['loc2'], errors='coerce')
distanceValues = distanceInfo.values
distanceValues = [tuple([int(x[0]), int(x[1]), x[2]]) for x in distanceValues.tolist()]
graph1 = nx.DiGraph()
graph1.add_weighted_edges_from(distanceValues)
pos = {1: (2, 6), 2: (5, 6), 3: (1, 5), 4: (3, 5), 5: (5, 5),  # 指定顶点位置
       6: (7, 5), 7: (2, 4), 8: (5, 4), 9: (1, 3), 10: (4, 3), 11: (6, 3),
       12: (7, 3), 13: (4, 2), 14: (6, 2), 15: (3, 1)}

start = {'A': 4, 'B': 6, 'C': 13}
end = {'1': 1, '2': 2, '3': 8, '4': 12, '5': 10, '6': 3, '7': 15, '8': 14}
others = [9, 5, 7, 11]
receive = ['start'] + list(end.keys())
distance_result = pd.DataFrame(columns=receive)
for s_key in start:
    temp = [s_key]
    for e_key in end:
        temp2 = nx.dijkstra_path_length(graph1, source=start[s_key], target=end[e_key])
        print('{}到{}的最短距离为：{}'.format(s_key, e_key, temp2))
        temp += [temp2]
    tempDf = pd.DataFrame(temp).T
    tempDf.columns = receive
    distance_result = pd.concat([distance_result, tempDf], axis=0)
distance_result.to_csv('distance_result.csv', index=False)
# for dr in distance_result.values:
# dr = ['A', 4.0, 8.0, 8.0, 19.0, 11.0, 6.0, 22.0, 20.0]
# dr = ['B', 14.0, 7.0, 7.0, 16.0, 12.0, 16.0, 23.0, 17.0]
dr = ['C', 20.0, 19.0, 11.0, 14.0, 6.0, 15.0, 5.0, 10.0]
start_i = dr[0]
end_s = list(end.values())
graph = nx.DiGraph()
graph.add_weighted_edges_from(distanceValues)
nx.draw(graph, pos, node_color='lime', node_size=200, arrows=False, with_labels=True, alpha=1)
labels = nx.get_edge_attributes(graph, 'weight')
path_list = getCellPath(graph, start[start_i], end_s)
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, arrows=False, font_color='black',
                             style='dashed')  # 显示权值
nx.draw_networkx_nodes(graph, pos, nodelist=[start[start_i]], node_color='red', node_size=200)  # 设置顶点颜色
nx.draw_networkx_nodes(graph, pos, nodelist=end_s, node_color='lime', node_size=200)  # 设置顶点颜色
nx.draw_networkx_nodes(graph, pos, nodelist=others, node_color='grey', node_size=210)  # 设置顶点颜色
nx.draw_networkx_edges(graph, pos, edgelist=path_list, edge_color='blue', arrows=True, arrowsize=20)  # 设置边的颜色
plt.savefig('fig/起点站{}各路径.png'.format(start_i))
graph.clear()