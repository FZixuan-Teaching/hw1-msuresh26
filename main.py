from hw1_simulate import *
import sys

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

  # Import contextlib for silencing output
  import contextlib, io
  @contextlib.contextmanager
  def silence():
      sys.stdout, old = io.StringIO(), sys.stdout
      try:
          yield
      finally:
          sys.stdout = old

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
  with silence():
    num_matched_greedy, num_patients_greedy = simulate(patients, donors, p_rate, d_rate, init_num_patients, init_num_donors, num_periods, usa_stats, compatible_blood_type, greedy_algorithm, rs);
    greedy_output = sys.stdout.getvalue()
  print("Number of matches from greedy algorithm: {:d}/{:d}".format(sum(num_matched_greedy.values()), num_patients_greedy))

  # Test integer linear programming approach
  print("\n====== MIP ======")
  rs.set_state(savedState)
  patients = {}
  donors = {}
  with silence():
    num_matched_mip, num_patients_mip = simulate(patients, donors, p_rate, d_rate, init_num_patients, init_num_donors, num_periods, usa_stats, compatible_blood_type, mip, rs);
    mip_output = sys.stdout.getvalue()
  print("Number of matches from MIP: {:d}/{:d}".format(sum(num_matched_mip.values()), num_patients_mip))

  # Test integer linear programming approach with a different objective function
  print("\n====== MIP2 ======")
  rs.set_state(savedState)
  patients = {}
  donors = {}
  with silence():
    num_matched_mip2, num_patients_mip2 = simulate(patients, donors, p_rate, d_rate, init_num_patients, init_num_donors, num_periods, usa_stats, compatible_blood_type, mip2, rs);
    mip2_output = sys.stdout.getvalue()
  print("Number of matches from MIP2: {:d}/{:d}".format(sum(num_matched_mip2.values()), num_patients_mip2))

  # Report greedy algorithm results
  print("\n====== Greedy ======")
  print(greedy_output)

  # Report integer linear programming approach results
  print("\n====== MIP ======")
  print(mip_output)

  # Report results from integer linear programming approach with a different objective function
  print("\n====== MIP2 ======")
  print(mip2_output)

# Add main function
if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run the blood donation simulation')
  parser.add_argument('--rs_seed', type=int, default=628, help='Random seed for simulation')
  args = parser.parse_args()
  main(args.rs_seed)
