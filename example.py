from hw1_simulate import *

# Create small example to visualize matches when using greedy vs mip
# # %pip install -q matplotlib
# # %pip install -q networkx
import networkx as nx
import matplotlib.pyplot as plt

rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(31415)))
example_num_patients = 5
example_num_donors = 5

patients = {'Patient '+str(key+1): generate_new(usa_stats, rs) for key in range(example_num_patients)}
donors   = {'Donor '+str(key+1): generate_new(usa_stats, rs) for key in range(example_num_donors)}

print("\n## Running greedy matching ##")
greedy_patient_status = {'Patient '+str(key+1): False for key in range(example_num_patients)}
greedy_donor_status   = {'Donor '+str(key+1): False for key in range(example_num_donors)}
greedy_matches = greedy_algorithm(patients, donors, greedy_patient_status, greedy_donor_status, compatible_blood_type)

print("\n## Running MIP matching ##")
mip_patient_status = {'Patient '+str(key+1): False for key in range(example_num_patients)}
mip_donor_status   = {'Donor '+str(key+1): False for key in range(example_num_donors)}
mip_matches = mip(patients, donors, mip_patient_status, mip_donor_status, compatible_blood_type)

print("Example matches: greedy/mip = {:d}/{:d}".format(len(greedy_matches),len(mip_matches)))
# display(mip_matches)
# display(greedy_matches)

# Draw compatibility graph
G = nx.Graph()
G.add_nodes_from(patients.keys(), bipartite=0)
G.add_nodes_from(donors.keys(), bipartite=1)

# Set positions of nodes
pos = nx.bipartite_layout(G, patients.keys())
for i in range(1, example_num_patients+1):
    pos['Patient '+str(i)] = (0, -i)
for i in range(1, example_num_donors+1):
    pos['Donor '+str(i)] = (1, -i)

# Add edges from patients to donors that are compatible, as dashed lines
for p in patients.keys():
  for d in donors.keys():
    if can_receive(patients[p], donors[d]):
      G.add_edge(p, d, style='dashed', color='red')

##### Optimized Matching #####
# Add edges for selected matches as thick solid blue lines
G.add_edges_from(mip_matches, style='solid', color='blue')

# Draw nodes, not filled in with any color, with a black solid border
nx.draw_networkx_nodes(G, pos, node_color='white', edgecolors='black', node_size=500)
# nx.draw_networkx_nodes(G, pos, nodelist=patients.keys(), node_size=500)
# nx.draw_networkx_nodes(G, pos, nodelist=donors.keys(), node_size=500)

# Draw edges
edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]
styles = [G[u][v]['style'] for u,v in edges]
nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=colors, style=styles)

# Draw labels to the left of nodes as 'P' or 'D' suffixed with number
nx.draw_networkx_labels(G, pos, labels={key: 'P'+key[-1] for key in patients.keys()}, font_size=10, font_family="sans-serif", font_color='black')
nx.draw_networkx_labels(G, pos, labels={key: 'D'+key[-1] for key in donors.keys()}, font_size=10, font_family="sans-serif", font_color='black')

# Draw legend
import matplotlib.lines as mlines
blue_line = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          markersize=10, label='Matched')

red_line = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=10, label='Compatible')
plt.legend(handles=[blue_line, red_line])

plt.title("Optimized Matching")
plt.axis('off')
#plt.show()
plt.savefig("optimized.jpg")

##### Greedy Matching #####
# Replace matched edges with greedy matches
G.remove_edges_from(mip_matches)
G.add_edges_from(mip_matches, style='dashed', color='red')
G.add_edges_from(greedy_matches, style='solid', color='blue')

# Draw graph
nx.draw_networkx_nodes(G, pos, node_color='white', edgecolors='black', node_size=500)
edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]
styles = [G[u][v]['style'] for u,v in edges]
nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=colors, style=styles)
nx.draw_networkx_labels(G, pos, labels={key: 'P'+key[-1] for key in patients.keys()}, font_size=10, font_family="sans-serif", font_color='black')
nx.draw_networkx_labels(G, pos, labels={key: 'D'+key[-1] for key in donors.keys()}, font_size=10, font_family="sans-serif", font_color='black')
plt.legend(handles=[blue_line, red_line])
plt.axis('off')
plt.title("Greedy Matching")
#plt.show()
plt.savefig("greedy.jpg")
