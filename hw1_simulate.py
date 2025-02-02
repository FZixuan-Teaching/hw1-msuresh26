# %%
import numpy as np
from gurobipy import *
import sys

# %%
## Set variables that will be used throughout

# Define which blood types are compatible with a patient
# For example, compatible_blood_type['A'] is 'A' and 'O'
# indicating that donors of type 'A' or type 'O' blood can donate to 'A' patients
compatible_blood_type = {
    'A':  ['A', 'O'], 
    'B':  ['B', 'O'], 
    'AB': ['A', 'B', 'AB', 'O'], 
    'O':  ['O']
}

# The stats dict will hold the probability of being every blood type
usa_stats = {
    'A':  .42, 
    'B':  .10, 
    'AB': .04, 
    'O':  .44
}
# Verify that probabilities sum to 1
assert(abs(1.-sum(usa_stats.values()) < 1e-7))

# %%
def generate_new(stats: dict, rs: np.random.RandomState) -> str:
  """Generate new patient or donor

  Parameters
  ----------
  stats : dict
    Probability of each blood type occurring
  """
  r = rs.uniform()
  sum_ = 0
  for bt in stats.keys():
    sum_ += stats[bt]
    if sum_ >= r:
      return bt


# %%
def can_receive(p: str, d: str, compatible_blood_type: dict = compatible_blood_type) -> bool:
  """Check compatibility of patient p to donor d

  Parameters
  ----------
  p : str
    Patient blood type
  d : str
    Donor blood type
  compatible_blood_type : dict
    For every patient blood type, this gives a list of donor blood types compatible with that patient
  """
  return d in compatible_blood_type[p]


# %%
## Greedy algorithm for transplants
def greedy_algorithm(P: dict, D: dict, patient_status: dict, donor_status: dict, compatible_blood_type: dict = compatible_blood_type):
  """Match donors and patients in a greedy fashion (no optimization)

  Parameters
  ----------
  P : dict
    Dictionary containing patient name as key (Patient X) and blood type as value
  D : dict
    Dictionary containing donor name as key (Donor Y) and blood type as value
  patient_status : dict
    Dictionary for whether patient has been matched (true / false)
  donor_status : dict
    Dictionary for whether donor has been matched (true / false)
  compatible_blood_type : dict
    For every patient blood type, this gives a list of donor blood types compatible with that patient
  """
  patients = [key for key in P.keys() if patient_status[key]==False]
  donors = [key for key in D.keys() if donor_status[key]==False]

  ### Question 1.1(b).i: Code the greedy algorithm and append matches to the list 'matches'
  matches = []
  for p in patients:
    patient_btype = P[p]
    for d in donors:
      if donor_status[d] == False and D[d] in compatible_blood_type[patient_btype]:
        patient_status[p] = True
        donor_status[d] = True
        matches.append((p, d))
        break

  return matches


# %%
## Integer linear programming approach for transplants
def mip(P: dict, D: dict, patient_status: dict, donor_status: dict, compatible_blood_type: dict = compatible_blood_type):
  """Match donors to patients based on optimization model

  Parameters
  ----------
  P : dict
    Dictionary containing patient name as key (Patient X) and blood type as value
  D : dict
    Dictionary containing donor name as key (Donor Y) and blood type as value
  patient_status : dict
    Dictionary for whether patient has been matched (true / false)
  donor_status : dict
    Dictionary for whether donor has been matched (true / false)
  compatible_blood_type : dict
    For every patient blood type, this gives a list of donor blood types compatible with that patient
  """
  patients = [key for key in P.keys() if patient_status[key]==False]
  donors = [key for key in D.keys() if donor_status[key]==False]

  ### Question 1.1(b).ii: Write your model down below and append matches to the list 'matches'
  model = Model('kidney-matching')
  sys.stdout.flush()

  # Variables: x_{i,j} binary representing whether patient i to donor j
  x = model.addVars(len(patients), len(donors), vtype=GRB.BINARY, name = "x")
  # Constraint: Each patient can be matched to at most one (compatible) donor
  for i, patient in enumerate(patients):
    model.addConstr(sum(x[i, j] for j in range(len(donors))) <= 1, name = f"numPatientMatches_{i},{patient}")
  # Constraint: Each donor can be matched to at most one (compatible) patient
  for j, donor in enumerate(donors):
    model.addConstr(sum(x[i,j] for i in range(len(patients))) <=1, name = f"numDonorMatches_{j},{donor}")

  for i, patient in enumerate(patients):
    for j, donor in enumerate(donors):
      if D[donor] not in compatible_blood_type[P[patient]]:
        model.addConstr(x[i,j]==0, name=f"incompatible_bt_{patient},{donor}")
  # Objective: Maximize number of transplants
  model.setObjective(sum(x[i,j] for i in range(len(patients)) for j in range(len(donors))), GRB.MAXIMIZE)
  # Optimize
  model.params.outputflag = 0
  model.optimize()
  model.params.LogToConsole = 0

  # Set matches based on solution to model
  matches = []
  if model.Status == GRB.OPTIMAL:
    for i in range(len(patients)):
      for j in range(len(donors)):
        if x[i,j].x == 1:
          matches.append((patients[i], donors[j]))

  return matches


# %%
## Second integer linear programming approach for transplants
def mip2(P: dict, D: dict, patient_status: dict, donor_status: dict, compatible_blood_type: dict = compatible_blood_type):
  """Match donors to patients based on optimization model

  Parameters
  ----------
  P : dict
    Dictionary containing patient name as key (Patient X) and blood type as value
  D : dict
    Dictionary containing donor name as key (Donor Y) and blood type as value
  patient_status : dict
    Dictionary for whether patient has been matched (true / false)
  donor_status : dict
    Dictionary for whether donor has been matched (true / false)
  compatible_blood_type : dict
    For every patient blood type, this gives a list of donor blood types compatible with that patient
  """
  patients = [key for key in P.keys() if patient_status[key]==False]
  donors = [key for key in D.keys() if donor_status[key]==False]

  ### Question 1.2: Write your model down below and append matches to the list 'matches'
  model = Model('modified-kidney-matching')

  # Variables: [fill in yourself]

  # Constraint: [fill in yourself]

  # Objective: [fill in yourself]

  # Optimize
  model.params.outputflag = 0
  model.optimize()
  model.params.LogToConsole = 0

  # Set matches based on solution to model
  matches = []

  return matches


# %%
def simulate(
    patients: dict, 
    donors: dict, 
    p_rate: float, 
    d_rate: float, 
    init_num_patients: int, 
    init_num_donors: int, 
    num_periods: int,
    stats: dict = usa_stats, 
    compatible_blood_type: dict = compatible_blood_type,
    match_function: callable = mip,
    rs: np.random.RandomState = np.random.RandomState(314),
    DEBUG: bool = False):
  """Simulate patient/donor pool over time

  Parameters
  ----------
  patients : dict
    Dictionary containing patient name as key (Patient X) and blood type as value
  donors : dict
    Dictionary containing donor name as key (Donor Y) and blood type as value
  p_rate : float
    Patients arrive at rate Poisson(p_rate)
  d_rate : float
    Donors arrive at rate Poisson(d_rate)
  init_num_patients : int
    Number of patients in system at start
  init_num_donors : int
    Number of donors in system at start
  num_periods : int
    Number of times patients+donors arrive and transplants are assigned
  stats : dict
    Frequency of each blood type in the population
  compatible_blood_type : dict
    For every patient blood type, this gives a list of donor blood types compatible with that patient
  match_function : function
    Function that returns a set of matches in the form of a list of pairs
  rs : random state from numpy
    Pass random generator to help reproducibility
  """

  # Set of patient / donors at start
  patients = {'Patient '+str(key+1): generate_new(stats, rs) for key in range(init_num_patients)}
  donors   = {'Donor '+str(key+1): generate_new(stats, rs) for key in range(init_num_donors)}

  # *_status[i] keeps whether patient/donor i has been matched
  patient_status = {'Patient '+str(key+1): False for key in range(init_num_patients)}
  donor_status   = {'Donor '+str(key+1): False for key in range(init_num_donors)}

  num_patients = init_num_patients
  num_donors = init_num_donors

  num_patients_by_type = {key: sum(patients[i] == key for i in patients) for key in compatible_blood_type.keys()}
  num_donors_by_type = {key: sum(donors[i] == key for i in donors) for key in compatible_blood_type.keys()}

  # To track avg number of patients matched by each blood type in each period
  num_matched_by_type = {key: 0 for key in compatible_blood_type.keys()}

  # To track time spent waiting for a transplant by each patient
  # and average time spent waiting for a transplant by blood type
  # Note that initializing with "0" skews statistics if number of periods is small
  TIS = {key: 0 for key in patients.keys()}

  # To track the average proportion of patients of each blood type that get matched per period
  num_avail_patients_by_type = {key: sum(patients[i] == key for i in patients) for key in compatible_blood_type.keys()}
  avg_prop_matched = {key: 0 for key in compatible_blood_type.keys()}

  curr_num_avail_patients = {key: sum(patients[i] == key for i in patients if patient_status[i] == False) for key in compatible_blood_type.keys()}
  curr_num_avail_donors = {key: sum(donors[i] == key for i in donors if donor_status[i] == False) for key in compatible_blood_type.keys()}
  if DEBUG: print("Init avail patients:", curr_num_avail_patients)
  if DEBUG: print("Init avail donors: ", curr_num_avail_donors)

  # print("Period progress bar: ", end="")
  for it in range(num_periods):
    # print(".", end="")
    if DEBUG: print("\nPeriod {:d}".format(it))
    # Generate new patients + donors
    new_patients = rs.poisson(p_rate)
    new_donors = rs.poisson(d_rate)

    new_patient_list = []
    new_donor_list = []
    for i in range(new_patients):
      # For each new patient, initialize time in system to 0
      curr_name = 'Patient '+str(num_patients+i+1)
      patients[curr_name] = generate_new(stats, rs)
      patient_status[curr_name] = False
      num_patients_by_type[patients[curr_name]] += 1
      num_avail_patients_by_type[patients[curr_name]] += 1
      TIS['Patient '+str(num_patients+i+1)] = 0
      new_patient_list.append(patients[curr_name])
    for i in range(new_donors):
      curr_name = 'Donor '+str(num_donors+i+1)
      donors[curr_name] = generate_new(stats, rs)
      donor_status[curr_name] = False
      new_donor_list.append(donors[curr_name])

    if DEBUG: print("New patients ({}):".format(new_patients), {key: sum(i == key for i in new_patient_list) for key in compatible_blood_type.keys()})
    if DEBUG: print("New donors ({}): ".format(new_donors), {key: sum(i == key for i in new_donor_list) for key in compatible_blood_type.keys()})

    num_patients += new_patients
    num_donors += new_donors

    curr_num_avail_patients = {key: sum(patients[i] == key for i in patients if patient_status[i] == False) for key in compatible_blood_type.keys()}
    curr_num_avail_donors = {key: sum(donors[i] == key for i in donors if donor_status[i] == False) for key in compatible_blood_type.keys()}
    if DEBUG: print("Num avail patients:", curr_num_avail_patients)
    if DEBUG: print("Num avail donors: ", curr_num_avail_donors)

    # Match donors and patients
    matches = match_function(patients, donors, patient_status, donor_status,
                             compatible_blood_type)

    # Update statistics: for every match m, 
    # m[0] (the patient) and m[1] (the donor) are no longer available or waiting
    curr_num_matched = {key: 0 for key in compatible_blood_type.keys()}
    for m in matches:
      # Double check compatibility
      assert(can_receive(patients[m[0]], donors[m[1]], compatible_blood_type))

      patient_status[m[0]] = True
      donor_status[m[1]] = True
      num_matched_by_type[patients[m[0]]] += 1
      curr_num_matched[patients[m[0]]] += 1
    
    if DEBUG: print("Num matched:", curr_num_matched)
    curr_num_avail_patients = {key: sum(patients[i] == key for i in patients if patient_status[i] == False) for key in compatible_blood_type.keys()}
    curr_num_avail_donors = {key: sum(donors[i] == key for i in donors if donor_status[i] == False) for key in compatible_blood_type.keys()}
    if DEBUG: print("Remaining avail patients:", curr_num_avail_patients)
    if DEBUG: print("Remaining avail donors: ", curr_num_avail_donors)

    # print("Period:", it)
    for key in compatible_blood_type.keys():
      curr_num_avail = num_avail_patients_by_type[key]

      curr_avg_prop_matched = 1. # TODO you need to calculate this value for the extra credit question (assume the proportion of matched patients is 1 when there are no patients of that type in a period)
      num_avail_patients_by_type[key] -= curr_num_matched[key]

    # For patients still in system, increment time spent waiting
    for i in patients.keys():
      if patient_status[i]==False:
        TIS[i] += 1

  # Report statistics
  avg_tis = 0 # TODO you need to calculate this value!
  TIS_BT = {key: 0 for key in compatible_blood_type.keys()}
  for i in patients.keys():
    TIS_BT[patients[i]] += TIS[i] / num_patients_by_type[patients[i]]

  print("=== Summary Statistics ===")
  print('Number of periods:', num_periods)
  print('Total # patients matched: {:d}/{:d}'.format(sum(num_matched_by_type.values()), num_patients))
  print('Number of patients by type:', num_patients_by_type)
  print('Number of patients matched:', num_matched_by_type)
  print('1.1(b)iii: Average number of patients (per blood type) matched per period:', {bt: num_matched_by_type[bt] / num_periods for bt in num_matched_by_type})
  print('1.1(b)iv: Average time in system:', sum(TIS.values())/ len(TIS))
  print('Average time in system (by type, weighed by num_patients):', {key : sum(TIS[i] for i in patients if patients[i] == key) / num_patients for key in compatible_blood_type.keys()})
  print('1.1(b)iv: Average time in system (by type, weighed by num_patients_by_type):', TIS_BT)
  print('1.1(b)v: Average proportion of patients matched per period by type:', "TODO for extra credit")

  return num_matched_by_type, num_patients

def main(rs_seed:int = 628):
  if (type(rs_seed) != int):
    raise TypeError("rs_seed must be an integer")

  # Define which blood types are compatible with a patient
  # For example, compatible_blood_type['A'] is 'A' and 'O'
  # indicating that donors of type 'A' or type 'O' blood can donate to 'A' patients
  compatible_blood_type = {
      'A':  ['A', 'O'],
      'B':  ['B', 'O'],
      'AB': ['A', 'B', 'AB', 'O'],
      'O':  ['O']
  }

  # The stats dict will hold the probability of being every blood type
  usa_stats = {
      'A':  .42,
      'B':  .10,
      'AB': .04,
      'O':  .44
  }
  # Verify that probabilities sum to 1
  assert(abs(1.-sum(usa_stats.values()) < 1e-7))

  # Initialize the simulation with some number of patients and donors
  init_num_patients = 20
  init_num_donors = 20

  # How long to run the simulation for
  num_periods = 52

  # Set the number of patients and donors arriving in each time period
  # The arrivals will be Poisson distributed with the below rate
  p_rate = 10
  d_rate = 10

  # Save random state
  from numpy.random import MT19937
  from numpy.random import RandomState, SeedSequence
  rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(rs_seed)))
  savedState = rs.get_state()

  # Test greedy algorithm
  print("====== Greedy ======")
  rs.set_state(savedState)
  patients = {}
  donors = {}
  num_matched_greedy, num_patients_greedy = simulate(patients, donors, p_rate, d_rate, init_num_patients, init_num_donors, num_periods, usa_stats, compatible_blood_type, greedy_algorithm, rs);
  print("Number of matches from greedy algorithm: {:d}/{:d}".format(sum(num_matched_greedy.values()), num_patients_greedy))

  # Test integer linear programming approach
  print("\n====== MIP ======")
  rs.set_state(savedState)
  patients = {}
  donors = {}
  num_matched_mip, num_patients_mip = simulate(patients, donors, p_rate, d_rate, init_num_patients, init_num_donors, num_periods, usa_stats, compatible_blood_type, mip, rs);
  print("Number of matches from MIP: {:d}/{:d}".format(sum(num_matched_mip.values()), num_patients_mip))

  # Test integer linear programming approach with a different objective function
  print("\n====== MIP2 ======")
  rs.set_state(savedState)
  patients = {}
  donors = {}
  num_matched_mip2, num_patients_mip2 = simulate(patients, donors, p_rate, d_rate, init_num_patients, init_num_donors, num_periods, usa_stats, compatible_blood_type, mip2, rs);
  print("Number of matches from MIP2: {:d}/{:d}".format(sum(num_matched_mip2.values()), num_patients_mip2))

# Add main function
if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run the blood donation simulation')
  parser.add_argument('--rs_seed', type=int, default=628, help='Random seed for simulation')
  args = parser.parse_args()
  main(args.rs_seed)

