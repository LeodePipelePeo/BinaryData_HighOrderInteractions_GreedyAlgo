//*** g++ -I/path/to/eigen -I/path/to/lbfgspp/include -O2 example.cpp ***/
//
//g++ -std=c++11 -I ./eigen3/ -I ./LBFGSpp-master/include -O2 main.cpp ReadData_DataBasisTransform.cpp IndepModel.cpp Models.cpp ModelStatistics.cpp BoltzmannLearning.cpp HeuristicAlgo.cpp BestBasis.cpp
//time ./a.out
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <list>
#include <map>
#include <random>  // for Mersen Twister  // also include "<cmath>" for function "log" (for instance)
#include <set>
#include <ctime> // for chrono
#include <ratio> // for chrono
#include <chrono> // for chrono
#include <bitset>
#include <string>

using namespace std;


/********************************************************************/
/**************************    CONSTANTS    *************************/
/********************************************************************/
#include "data.h"

/******************************************************************************/
/**************************     READ FILE    **********************************/
/******************************************************************************/
map<uint32_t, unsigned int> read_datafile(unsigned int *N, string filename = datafile);   // O(N)  //N = data set size  //READ DATA and STORE them in Nset


/******************************************************************************/
/************************     CHANGE of BASIS      ****************************/
/***************************    Build K_SET     *******************************/
/******************************************************************************/
map<uint32_t, unsigned int> build_Kset(map<uint32_t, unsigned int> Nset, list<uint32_t> Basis_SCModel, bool print_bool=false);
map<uint32_t, unsigned int> build_cut_Kset(map<uint32_t, unsigned int> Kset, uint32_t community);
unsigned int k1_op(map<uint32_t, unsigned int> Nset, uint32_t op);
// mu_m = states of the systems written in the basis specified in `list<uint32_t> Basis_SCModel`
// Kset[sig_m] = Number of times the state mu_m appears in the transformed data set
//
// Rem: the new basis can have a lower dimension then the original dataset; in which case the function
// will reduce the dataset to the subspace defined by the specified basis.

// Build Kset for the states written in the basis of the m-chosen independent
// operator on which the SC model is based:

uint32_t state_cut(uint32_t state, uint32_t part);

/******************************************************************************/
/**************************   MODEL TYPE   *********************************/
/******************************************************************************/
double Ising(Interaction I, uint32_t state, int *Op_s);   // s_i in {-1;1}   !! Convention !! {0,1} in data <-> {1,-1} in model
double ACE(Interaction I, uint32_t state, int *Op_s);    // s_i in {0;1}    !! Convention !! {0,1} in data <-> {0,1} in model; But mapping {0,1} <-> {-1,1} in Ising

/******************************************************************************/
/************************** INDEPENDENT MODEL *********************************/
/******************************************************************************/
list<Interaction> IndepModel_fromBasis(vector<pair<Operator, bool> > best_basis);

/************************** Probability space *********************************/
void IndepModel_VS_data_ProbaSpace(vector<pair<Operator, bool> > best_basis, map<uint32_t, unsigned int> Nset_test, unsigned int N_test, string outputfilename, unsigned int variables = n);
double* Probability_AllStates_Indep(vector<pair<Operator, bool> > best_basis, unsigned int variables = n);

/************************** Fourier space *************************************/
set<Operator> Rank_m1_Indepmodel(set<Operator> allOp, double *P_indep, unsigned int variables = n);
set<Operator> Rank_m1_Model(set<Operator> allOp, double *P, double Z, unsigned int variables = n);

/******************************** Noise Model *********************************/
list<Interaction> Noise_Model(vector< pair<uint32_t,double> > IndepModel);

/******************************************************************************/
/***************************  ADD NEXT INTERACTION  ***************************/
/******************************************************************************/
Interaction Next_Model(set<Operator> &m1_ranked, list<Interaction> &list_I, map<uint32_t, unsigned int> Nset, unsigned int N, double *L, vector<uint32_t> communities, bool final_run, bool *stop, unsigned int variables = n);

/******************************************************************************/
/*************************  MODEL: LIST of INTERACTIONS ***********************/
/******************************************************************************/

/**************************** OTHER MODELS ************************************/
list<Interaction> FullyConnectedPairwise();
list<Interaction> Complete_Model();
list<Interaction> IndepModel(map<uint32_t, unsigned int> Nset, unsigned int N, unsigned int variables = n);


void all_int_kbits(int k);

/******************************* ANY MODEL ************************************/
list<Interaction> Build_Model(uint32_t Op_M[], int K);

/************************* PRINT MODEL in TERM ********************************/
void PTerm_ListInteraction (list<Interaction> list_I);

/************************* PRINT MATRIX PAIRWISE INTERACTION ********************************/
void Print_matrix_J_pairwise(string F_name, list<Interaction> list_I);

/******************************************************************************/
/************* PARTITION FUNCTION and PROBA of ALL STATES *********************/
/******************************************************************************/
// return the probability (not normalised) of all the states and compute the partition function
double* Probability_AllStates(double MConvention(Interaction, uint32_t, int*), list<Interaction> list_I, double *Z, unsigned int variables = n);

/************************** PRINT in FILE PROBA *******************************/
// model probability of all the states:
void print_proba_state(double* P, double Z, string OUTPUTfilename, unsigned int variables = n);
// model probability VS data frequencies of the observed states:
void Model_VS_data_ProbaSpace(double *P, double Z, map<uint32_t, unsigned int> Nset, unsigned int N_test, string outputfilename, unsigned int variables = n);

/************************ PRINT in FILE FOURIER SPACE *************************/
void All_Op_averages_Ising(map<uint32_t, unsigned int> Nset, double *P, double Z, unsigned int N, string outputfilename, unsigned int variables = n);
void print_m1_Data_VS_Model(map<uint32_t, unsigned int> Nset, double* Op_m1_Model, unsigned int N, string outputfilename);

/******************************************************************************/
/****************************    LOG-LIKELIHOOD    ****************************/
/******************************************************************************/
double LogLikelihood(double *P, double Z, map<uint32_t, unsigned int> Nset);
double LogL_bis(list<Interaction> list_I, double Z, unsigned int N);
double LogL(double MConvention(Interaction, uint32_t, int*), list<Interaction> list_I, unsigned int N, unsigned int variables=n);

double LogLikelihood_CompleteModel(map<uint32_t, unsigned int> Nset, unsigned int N);
/******************************************************************************/
/**************************** Operator Averages *******************************/
/******************************************************************************/
void Model_averages(double MConvention(Interaction, uint32_t, int*), double *P, double Z, list<Interaction> &list_I, unsigned int N);

void empirical_averages_Ising(map<uint32_t, unsigned int> Nset, list<Interaction> &list_I, unsigned int N);
void empirical_averages_ACE(map<uint32_t, unsigned int> Nset, list<Interaction> &list_I, unsigned int N);

double* AllOp_av(double *P, double Z, unsigned int N);  // _all_Model_averages in ISING convention
double* AllOp_m1(double *P, double Z, unsigned int N);

void Print_Correlations(map<uint32_t, unsigned int> Nset, unsigned int N);

/******************************************************************************/
/******************************   Ranked m1   *********************************/
/******************************************************************************/
set<Operator> AllOp_m1_data_rank(map<uint32_t, unsigned int> Nset, unsigned int N, unsigned int variables = n);

void Fill_m1_model(set<Operator> &allOp, double *P, double Z);

//set<Operator> AllOp_m1_Model_ranked(double *P, double Z, unsigned int N);

//void print_m1_Data_VS_Model_Mranked(map<uint32_t, unsigned int> Nset, set<Operator> Op_m1_Model, unsigned int N, string outputfilename);
void print_m1_Data_VS_Model_DKLranked(set<Operator> Op_m1_Model, unsigned int N, string outputfilename);

/******************************************************************************/
/******************************   Pairwise model   ****************************/
/******************************************************************************/
set<Operator> PairOp_m1_data_rank(map<uint32_t, unsigned int> Nset, unsigned int N);

/******************************************************************************/
/****************************  Best Basis  ************************************/
/******************************************************************************/
vector<pair<Operator, bool> > select_best_basis(set<Operator> &allOp_data, double Nd, double *LogL);
void printfile_BestBasis(vector<pair<Operator, bool> > Op, double Nd, string name);

/******************************************************************************/
/**************************** Boltzmann Learning ******************************/
/******************************************************************************/
double BoltzmannLearning_Ising(map<uint32_t, unsigned int> Nset, list<Interaction> &list_I, unsigned int N, unsigned int variables=n);
double BoltzmannLearning_ACE(map<uint32_t, unsigned int> Nset, list<Interaction> &list_I, unsigned int N);

void gIsing_from_gACE(list<Interaction> &list_I);

/******************************************************************************/
/****************************  Analysed Data  *********************************/
/******************************************************************************/
void USSC_BestModel_all_K45(map<uint32_t, unsigned int>  Nset, unsigned int N);

/******************************************************************************/
/****************************  BEST INDEP MODEL *******************************/
/******************************************************************************/
set<Operator> Indep_Model_For_MCM(map<uint32_t, unsigned int> Nset, unsigned int N, list<Interaction> &list_I, double *L, string OUTPUT_filename, unsigned int variables = n)
{
  double Nd = (double) N;

  cout << endl << "--->> Search Best Basis.. \t";
  set<Operator> m1_ranked = AllOp_m1_data_rank(Nset, N, variables);
  cout << "Total number of Operators = " << m1_ranked.size() << endl << endl;
  print_m1_Data_VS_Model_DKLranked(m1_ranked, N, OUTPUT_filename + "_FourierSpace_dataVSNoise_DKLranked.dat");

  if (list_I.size() == 0)
  {
    list_I = IndepModel(Nset, N, variables);
  }

  PTerm_ListInteraction (list_I);

/// This function is updating the layer values: Rem, still an issue with the layers values,
/// as we remove the basis op from the "m1_Indep_ranked" while searching for the best basis:

  (*L)=0;
  double S=0;

  //vector<pair<Operator, bool> > best_basis = select_best_basis(m1_ranked, ((double) N), L);
  vector<pair<Operator, bool>> basis;
  Operator Op;
  list<Interaction>::iterator it;
  for (it = list_I.begin(); it != list_I.end(); it++)
  {
    Op.bin=(*it).Op;
    Op.p1_D=k1_op(Nset, Op.bin)/Nd;
    S = -Op.p1_D * log(Op.p1_D) - (1-Op.p1_D)*log(1-Op.p1_D); // - [ p1*log(p1) + (1-p1)*log(1-p1) ]
    (*L) += - Nd*S;
    basis.push_back(make_pair(Op, false));
  }

  /*
  printfile_BestBasis(basis, (double) N, "");

  //Print BIC:
  fstream BIC_file( (OUTPUT_filename + "_BestIndep_toK"+ to_string(n) +".dat").c_str(), ios::out);
  BIC_file << "##1:K \t 2:maxlogL \t 3: BIC \t 4:worst_MDL \t 5:Op" << endl << "##" << endl;

  double L_buff = -Nd * n * log(2.);
  for (vector<pair<Operator, bool> >::iterator Op = basis.begin(); Op != basis.end(); Op++)
  {
    K++;       BIC_file << K << "\t";
    L_buff += ( - Nd * (*Op).first.S + Nd * log(2.) );
    BIC_file << L_buff << "\t";
    BIC_file << L_buff - K * log( Nd /(2.*M_PI)) / 2. << "\t";   //BIC
    BIC_file << L_buff - K * log( (Nd*M_PI) /2.) / 2. << "\t";   //MDL worst
    BIC_file << (*Op).first.bin << "\t" << bitset<n>((*Op).first.bin) << endl;
  }
  BIC_file.close();
  */

  int K=0;

  K = list_I.size();
  cout << "Number of parameters, K =  " << K << endl;
  cout << "Max Log-Likelihood, L = " << (*L) << endl;
  cout << "BIC: " << (*L) - K * log( (Nd) /(2.*M_PI)) / 2. <<  endl << endl;

  // Print all probabilities:
  double* P_indep = Probability_AllStates_Indep(basis, variables); // Proba each state (normalised)
  IndepModel_VS_data_ProbaSpace(basis, Nset, N, OUTPUT_filename, variables);

  // Indep model all m1:
  cout << "Model Rank all DKL.." << endl;
  set<Operator> allOp_buffer = Rank_m1_Indepmodel(m1_ranked, P_indep, variables);
  m1_ranked.clear();
  //m1_ranked.swap(allOp_buffer);

  print_m1_Data_VS_Model_DKLranked(m1_ranked, N, OUTPUT_filename + "_FourierSpace_dataVSIndep_DKLranked.dat");

  return allOp_buffer;
}

set<Operator> best_Indep_PairwiseModel(map<uint32_t, unsigned int> Nset, unsigned int N, list<Interaction> &list_I, double *L, string OUTPUT_filename)
{
  double Nd = (double) N;
  int n_int = n;

  cout << endl << "--->> Search Best Basis.. \t";
  set<Operator> m1_ranked = PairOp_m1_data_rank(Nset, N);
  cout << "Total number of Operators = " << m1_ranked.size() << endl << endl;
  print_m1_Data_VS_Model_DKLranked(m1_ranked, N, OUTPUT_filename + "_Pair_FourierSpace_dataVSNoise_DKLranked.dat");

/// This function is updating the layer values: Rem, still an issue with the layers values,
/// as we remove the basis op from the "m1_Indep_ranked" while searching for the best basis:
  vector<pair<Operator, bool> > best_basis = select_best_basis(m1_ranked, ((double) N), L);
  printfile_BestBasis(best_basis, Nd, "Pair_");

  //Print BIC:
  fstream BIC_file( (OUTPUT_filename + "_Pair_BestIndep_toK"+ to_string(n) +".dat").c_str(), ios::out);
  BIC_file << "##1:K \t 2:maxlogL \t 3: BIC \t 4:worst_MDL \t 5:Op" << endl << "##" << endl;

  int K=0; double L_buff = -Nd * n * log(2.);
  for (vector<pair<Operator, bool> >::iterator Op = best_basis.begin(); Op != best_basis.end(); Op++)
  {
    K++;       BIC_file << K << "\t";
    L_buff += ( - Nd * (*Op).first.S + Nd * log(2.) );
    BIC_file << L_buff << "\t";
    BIC_file << L_buff - K * log( Nd /(2.*M_PI)) / 2. << "\t";   //BIC
    BIC_file << L_buff - K * log( (Nd*M_PI) /2.) / 2. << "\t";   //MDL worst
    BIC_file << (*Op).first.bin << "\t" << bitset<n>((*Op).first.bin) << endl;
  }
  BIC_file.close();

  list_I = IndepModel_fromBasis(best_basis);
  PTerm_ListInteraction (list_I);

  K = list_I.size();
  cout << "Number of parameters, K =  " << K << endl;
  cout << "Max Log-Likelihood, L = " << (*L) << endl;
  cout << "BIC: " << (*L) - K * log( ((double) N) /(2.*M_PI)) / 2. <<  endl << endl;

  // Print all probabilities:
  double* P_indep = Probability_AllStates_Indep(best_basis, n_int); // Proba each state (normalised)
  IndepModel_VS_data_ProbaSpace(best_basis, Nset, N, OUTPUT_filename + "_Pair_", n_int);

  // Indep model all m1:
  cout << "Model Rank all DKL.." << endl;


  set<Operator> allOp_buffer = Rank_m1_Indepmodel(m1_ranked, P_indep, n_int);
  m1_ranked.clear();
  //m1_ranked.swap(allOp_buffer);

  print_m1_Data_VS_Model_DKLranked(m1_ranked, N, OUTPUT_filename + "_Pair_FourierSpace_dataVSIndep_DKLranked.dat");

  return allOp_buffer;
}


/******************************************************************************/
/*************************  HEURISTIC ALGORITHM *******************************/
/******************************************************************************/
void Heuristic_Model_K(map<uint32_t, unsigned int> Nset, unsigned int N, list<Interaction> &list_I, set<Operator> &m1_ranked, string OUTPUT_filename, vector<uint32_t> communities, bool final_run, int Kmax = n)
{
  cout << endl << "--->> Search next most relevant Operators.. \t";
  cout << "Number of Operators left = " << m1_ranked.size() << endl << endl;

  //variables:
  int K = list_I.size();
  set<Operator> allOp_buffer;
  double L=0;  // max-logLikelihood
  double Z=0;  // partition function
  double *P;
  double *m1;
  double Nd = (double) N;
  bool stop = false;

  //Print BIC:
  fstream BIC_file( (OUTPUT_filename + "_Best_toK"+ to_string(Kmax) +".dat").c_str(), ios::out);
  BIC_file << "##1:K \t 2:maxlogL \t 3: BIC \t 4:worst_MDL \t 5:Op" << endl << "##" << endl;

  while(K < Kmax)
  {
    K += 1;
    Interaction I_K = Next_Model(m1_ranked, list_I, Nset, N, &L, communities, final_run, &stop);
    PTerm_ListInteraction (list_I);

    BIC_file << K << "\t";
    BIC_file << L << "\t";
    BIC_file << L - K * log( Nd /(2.*M_PI)) / 2. << "\t";
    BIC_file << L - K * log( (Nd*M_PI) /2.) / 2. << "\t";
    BIC_file << I_K.Op << "\t" << bitset<n>(I_K.Op) << endl;

    // UPDATE p1 and DKL:
    P = Probability_AllStates(Ising, list_I, &Z); // Proba each state, non normalised

    //P = Probability_AllStates(ACE, list_I, &Z); // Proba each state, non normalised // ACE version
    allOp_buffer = Rank_m1_Model(m1_ranked, P, Z);
    m1_ranked.clear();
    m1_ranked.swap(allOp_buffer);

    //PRINT OUT:

    // LEON ADDED
    All_Op_averages_Ising(Nset, P, Z, N, OUTPUT_filename + "_phi_K"+to_string(K)+".dat");

    //cout << "--->> Print file: Best model, Fourier Space, m1 of all the states.." << endl;
    print_proba_state(P, Z, OUTPUT_filename + "_AllProba_K"+to_string(K)+".dat");
    Model_VS_data_ProbaSpace(P, Z, Nset, N, OUTPUT_filename + "_ProbaSpace_K"+to_string(K)+".dat");// model probability VS data frequencies of the observed states
    print_m1_Data_VS_Model_DKLranked(m1_ranked, N, OUTPUT_filename + "_FourierSpace_dataVSK" + to_string(K) + "_DKLranked.dat");
    //m1 = AllOp_m1(P, Z, N);

    //print_m1_Data_VS_Model(Nset, m1, N, OUTPUT_filename + "_FourierSpace_dataVSK"+to_string(K)+".dat");

    cout << "Number of Operators left = " << m1_ranked.size() << endl << endl;
  }
  BIC_file.close();
}

void Heuristic_Model_K_MCM(map<uint32_t, unsigned int> Nset, unsigned int N, list<Interaction> &list_I, set<Operator> &m1_ranked, string OUTPUT_filename, vector<uint32_t> communities, bool final_run, int Kmax = n, unsigned int variables = n)
{
  cout << endl << "--->> Search next most relevant Operators.. \t";
  cout << "Number of Operators left = " << m1_ranked.size() << endl << endl;

  //variables:
  int K = list_I.size();

  set<Operator> allOp_buffer;
  double L=0;  // max-logLikelihood
  double Z=0;  // partition function
  double *P;
  double *m1;
  double Nd = (double) N;
  bool stop = false;

  //Print BIC:
  fstream BIC_file( (OUTPUT_filename + "_Best_toK"+ to_string(Kmax) +".dat").c_str(), ios::out);
  BIC_file << "##1:K \t 2:maxlogL \t 3: BIC \t 4:worst_MDL \t 5:Op" << endl << "##" << endl;

  while(K < Kmax)
  {
    K += 1;
    Interaction I_K = Next_Model(m1_ranked, list_I, Nset, N, &L, communities, final_run, &stop, variables);


    if (stop == false)
    {
      PTerm_ListInteraction (list_I);

      BIC_file << K << "\t";
      BIC_file << L << "\t";
      BIC_file << L - K * log( Nd /(2.*M_PI)) / 2. << "\t";
      BIC_file << L - K * log( (Nd*M_PI) /2.) / 2. << "\t";
      BIC_file << I_K.Op << "\t" << bitset<n>(I_K.Op) << endl;

      // UPDATE p1 and DKL:
      P = Probability_AllStates(Ising, list_I, &Z, variables); // Proba each state, non normalised

      //P = Probability_AllStates(ACE, list_I, &Z); // Proba each state, non normalised // ACE version
      allOp_buffer = Rank_m1_Model(m1_ranked, P, Z, variables);
      m1_ranked.clear();
      m1_ranked.swap(allOp_buffer);

      //PRINT OUT:

      // LEON ADDED
      All_Op_averages_Ising(Nset, P, Z, N, OUTPUT_filename + "_phi_K"+to_string(K)+".dat", variables);

      //cout << "--->> Print file: Best model, Fourier Space, m1 of all the states.." << endl;
      print_proba_state(P, Z, OUTPUT_filename + "_AllProba_K"+to_string(K)+".dat", variables);
      Model_VS_data_ProbaSpace(P, Z, Nset, N, OUTPUT_filename + "_ProbaSpace_K"+to_string(K)+".dat", variables);// model probability VS data frequencies of the observed states
      print_m1_Data_VS_Model_DKLranked(m1_ranked, N, OUTPUT_filename + "_FourierSpace_dataVSK" + to_string(K) + "_DKLranked.dat");
      //m1 = AllOp_m1(P, Z, N);

      //print_m1_Data_VS_Model(Nset, m1, N, OUTPUT_filename + "_FourierSpace_dataVSK"+to_string(K)+".dat");

      cout << "Number of Operators left = " << m1_ranked.size() << endl << endl;
    }
    else
    {
      cout << "Algorithm stopped because an interaction within a community was next to be added" << endl;
      break;
    }

  }
  free(P);
  BIC_file.close();
}



uint32_t operators_back_to_n(uint32_t old_operator, uint32_t community)
{

  uint32_t new_operator = 0;
  int old_position = 0;
  uint32_t variable = 1;

  for (int var_pos = 0; var_pos < n; var_pos++)
  {
    if (bitset<n>(community)[var_pos] == 1)
    {
      if (bitset<n>(old_operator)[old_position] == 1)
      {
        new_operator = new_operator + (variable << var_pos);
        variable = 1;
      }

      old_position++;
    }
  }

  return new_operator;
}



/******************************************************************************/
/************************** MAIN **********************************************/
/******************************************************************************/
//Rem : ---> 2^30 ~ 10^9
int main(int argc, char **argv)
{
  //*******************************************
  //********** READ DATA FILE:  ***************     -->  data are put in Nset:
  //*******************************************    // Nset[mu] = # of time state mu appears in the data set
  // Choice of the basis for building the Mimimally Complex Model (MCM):
  uint32_t Basis_Choice_USSC[] = {36, 10, 3, 272, 260, 320, 130, 65, 4};    // Ex. This is the best basis for the USSC dataset
  uint32_t Basis_Choice[] = {2064, 10240, 8196, 8448, 24, 96, 1032, 1152, 1024, 4608, 8194, 257, 4128, 80};

  //int Kmax_list[] = {15, 2, 2, 2, 1, 1, 1};
  int Kmax_list[] = {15, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  // All communities
  vector<uint32_t> communities;
  /*
  uint32_t com_1 = 440;
  uint32_t com_2 = 70;
  uint32_t com_3 = 1;

  communities.push_back(com_1);
  communities.push_back(com_2);
  communities.push_back(com_3);
  */

  /*
  uint32_t com_1 = 31;
  uint32_t com_2 = 4128;
  uint32_t com_3 = 320;
  uint32_t com_4 = 1152;
  uint32_t com_5 = 512;
  uint32_t com_6 = 2048;
  uint32_t com_7 = 8192;
  */

  uint32_t com_1 = 31;
  uint32_t com_2 = 32;
  uint32_t com_3 = 64;
  uint32_t com_4 = 128;
  uint32_t com_5 = 256;
  uint32_t com_6 = 512;
  uint32_t com_7 = 1024;
  uint32_t com_8 = 2048;
  uint32_t com_9 = 4096;
  uint32_t com_10 = 8192;

  communities.push_back(com_1);
  communities.push_back(com_2);
  communities.push_back(com_3);
  communities.push_back(com_4);
  communities.push_back(com_5);
  communities.push_back(com_6);
  communities.push_back(com_7);
  communities.push_back(com_8);
  communities.push_back(com_9);
  communities.push_back(com_10);


  int amount_of_communities = communities.size();
  vector<unsigned int> com_count;
  for (int j = 0; j < amount_of_communities; j++)
  {
    com_count.push_back(bitset<n>(communities[j]).count());
  }


  unsigned int m = sizeof(Basis_Choice) / sizeof(uint32_t);
  list<uint32_t> Basis_li;  Basis_li.assign (Basis_Choice, Basis_Choice + m);

  // Add default basis:

  cout << endl << "*******************************************************************************************";
  cout << endl << "****************  TRAINING = TEST on the WHOLE dataset:  **********************************";
  cout << endl << "*******************************************************************************************" << endl;

  cout << endl << "*******************************************************************************************";
  cout << endl << "***********************************  Read the data:  **************************************";
  cout << endl << "*******************************************************************************************" << endl;

  string filename;
  if (argc > 1) {    filename = argv[1];  }  // output file name
  else {    cout << "The execution is missing the input filename as an argument." << endl;  }

  unsigned int N = 0; // will contain the number of datapoints in the dataset
  map<uint32_t, unsigned int> Nset = read_datafile(&N, filename);
  //cout << "N = " << N << endl;

  cout << endl << "*******************************************************************************************";
  cout << endl << "*********************************  Change the data basis   ********************************";
  cout << endl << "**************************************  Build Kset:  **************************************";
  cout << endl << "*******************************************************************************************" << endl;
  // Transform the data in the specified in Basis_SCModel[];
  map<uint32_t, unsigned int> Kset = build_Kset(Nset, Basis_li, false);

  //Print info SC model:
  int i = 1;
  for (list<uint32_t>::iterator it = Basis_li.begin(); it != Basis_li.end(); it++)
  {
    cout << "##\t " << i << " \t " << (*it) << " \t " << bitset<n>(*it) << endl; i++;
  } cout << "##" << endl;


  cout << "--->> Create OUTPUT Folder: (if needed) ";
  system( ("mkdir " + OUTPUT_directory).c_str() );
  cout << endl;



  // ************************************************************************************************** //


  /********************************** CUTTING THE DATA **********************************************/

  cout << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" << endl;

  // state repair test
  cout << "state back to n variables test" << endl;

  uint32_t test_community = 103;
  uint32_t test_operator = 21;

  cout << "test uint32_t = " << test_community << " to bitset = " << bitset<n>(test_community) << endl << endl;

  uint32_t new_test_op = operators_back_to_n(test_operator, test_community);
  cout << "test operator = " << bitset<n>(test_operator) << "\ttest community = " << bitset<n>(test_community) << "\tnew operator = " << bitset<n>(new_test_op) << endl;

  cout << endl << endl;
  // state cut test
  cout << "state cut test" << endl;
  uint32_t new_state;
  unsigned int frequency;
  uint32_t state = bitset<n>("000001000").to_ulong();
  uint32_t part = bitset<n>("000001010").to_ulong();
  cout << "state = " << bitset<n>(state) << "\tpart = " << bitset<n>(part) << endl;
  new_state = state_cut(state, part);
  cout << "new state = " << bitset<n>(new_state) << endl;



  /*
  cout << "SMALLER KSETS TEST" << endl;
  int count_frequencies = 0;
  map<uint32_t, unsigned int>::iterator it;
  cout << "Community = " << bitset<n>(communities[i]) << endl;

  for (it = Kset_cut.begin(); it != Kset_cut.end(); it++)
  {
    cout << "Kset #" << i << ": state = " << it->first << "\tfrequency = " << it->second << endl;
    count_frequencies += it->second;
  }
  cout << "Frequency count = " << count_frequencies << endl;
  */

  cout << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" << endl;








  string OUTPUT_filename = OUTPUT_directory + filename;
  //OUTPUT_filename_Data_VS_Model = OUTPUT_directory + filename + "_dataVSmodel.dat";

  // cutting Kset
  map<uint32_t, unsigned int> Kset_cut;

  // Kset_cut loop
  list<Interaction> list_total_I;
  list<Interaction> list_I;
  Interaction I;
  int Kmax = 20;
  int K=0;
  double L=0;
  double Nd = (double) N;
  unsigned int ROp_tot;
  bool final_run = false;
  list<Interaction>::iterator it2;
  double p1 = 0;

  for (int i = 0; i < communities.size(); i++)
  {

    if (com_count[i] > 1)
    {

      //cout << "--->> Print file: Correlations in Ising C1 convention.." << endl;
      //Print_Correlations(Nset, N);

      Kset_cut = build_cut_Kset(Kset, communities[i]);

      K=0; L=0;

      cout << endl << "***********************************  Complete model for communities[" << i << "]:  *************************************" << endl;

      ROp_tot = (1 << com_count[i]) - 1;

      K = ROp_tot;
      L = LogLikelihood_CompleteModel(Kset_cut, N);

      cout << "Number of parameters, K =  " << K << endl;
      cout << "Max Log-Likelihood, L = " << L << endl;
      cout << "BIC: " << L - K * log( Nd /(2.*M_PI)) / 2. <<  endl << endl;

    //  cout << endl << "*****************************  Best Independent model:  **********************************" << endl;

      set<Operator> m1_ranked_MCM = Indep_Model_For_MCM(Kset_cut, N, list_I, &L, OUTPUT_filename, com_count[i]);
      // UPDATE p1 and DKL:

      Heuristic_Model_K_MCM(Kset_cut, N, list_I, m1_ranked_MCM, OUTPUT_filename + "community[" + to_string(i) + "]", communities, final_run , Kmax_list[i], com_count[i]);

      //check
      Kset_cut.clear();

    }
    else
    {

      I.Op = communities[i];
      I.k = bitset<n>(I.Op).count();


      p1 = ((double) k1_op(Kset, I.Op)) / Nd;
      I.g_Ising = 0.5*log( (1.- p1) / p1 );

      I.g = I.g_Ising;  I.g_ACE = 0;
      I.av_D = 0; I.av_M = 0;

      list_I.push_back(I);

    }


    /*
    cout << "list change test 110111000WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW" << endl;

    cout << "Before change AAAAAAAAAA" << endl;
    PTerm_ListInteraction(list_I);
    */

    // Operators back to n variables
    if (com_count[i] > 1)
    {
      for (it2 = list_I.begin(); it2 != list_I.end(); it2++)
      {
        (*it2).Op = operators_back_to_n((*it2).Op, communities[i]);
      }
    }

    /*
    cout << "After change AAAAAAAAAA" << endl;
    PTerm_ListInteraction(list_I);
    */
    /*
    cout << "LIST I 0000000000000000000000000000000000000000000000000000" << endl;
    PTerm_ListInteraction(list_I);
    */

    list_total_I.splice(list_total_I.end(), list_I);

    /*
    cout << "LIST TOTAL I 0000000000000000000000000000000000000000000000000000" << endl;
    PTerm_ListInteraction(list_total_I);
    */
    list_I.clear();




  }
  /*
  cout << "List total test AAAAAAAAAA" << endl;
  PTerm_ListInteraction(list_total_I);
  */

  cout << "CCCCCCCCCCCCCCCCCCCC FINAL RUN CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"<< endl;
  set<Operator> allOp_buffer;
  double Z;
  double *P;
  final_run = true;
  set<Operator> m1_ranked_final = Indep_Model_For_MCM(Kset, N, list_total_I, &L, OUTPUT_filename, n);
  P = Probability_AllStates(Ising, list_total_I, &Z); // Proba each state, non normalised
  //P = Probability_AllStates(ACE, list_I, &Z); // Proba each state, non normalised // ACE version
  allOp_buffer = Rank_m1_Model(m1_ranked_final, P, Z);
  m1_ranked_final.clear();
  m1_ranked_final.swap(allOp_buffer);


  // UPDATE p1 and DKL:
  Kmax = 100;
  Heuristic_Model_K_MCM(Kset, N, list_total_I, m1_ranked_final, OUTPUT_filename + "_final", communities, final_run, Kmax, n);


//  cout << "--->> Print file: Correlations in Ising C1 convention.." << endl;
//  Print_Correlations(Nset, N);


  /*
  int K=0; double L=0;
  double Nd = (double) N;

  cout << endl << "***********************************  Complete model:  *************************************" << endl;

  unsigned int ROp_tot = (1 << com_count[i]) - 1;

  K = ROp_tot;
  L = LogLikelihood_CompleteModel(Kset_cut, N);

  cout << "Number of parameters, K =  " << K << endl;
  cout << "Max Log-Likelihood, L = " << L << endl;
  cout << "BIC: " << L - K * log( Nd /(2.*M_PI)) / 2. <<  endl << endl;

  list<Interaction> list_I;

//  cout << endl << "*****************************  Best Independent model:  **********************************" << endl;

  set<Operator> m1_ranked_MCM = Indep_Model_For_MCM(Kset_cut, N, list_I, &L, OUTPUT_filename, communities, com_count[i]);
  int Kmax = 20;
  Heuristic_Model_K_MCM(Kset_cut, N, list_I, m1_ranked_MCM, OUTPUT_filename, communities, final_run, Kmax, com_count[i]);
  */




  /*

  cout << endl << "*****************************  Best model with K iterations:  **********************************" << endl;
  int Kmax = 10;
  set<Operator> m1_ranked = best_Indep_Model(Nset, N, list_I, &L, OUTPUT_filename);
  //set<Operator> m1_ranked = AllOp_m1_data_rank(Nset, N);
  Heuristic_Model_K(Kset, N, list_I, m1_ranked, OUTPUT_filename, communities, final_run, Kmax);

  */

//  cout << endl << "*****************************  Best independent Model, among Pairwise models:  **********************************" << endl;
//  set<Operator> m1_ranked = best_Indep_PairwiseModel(Nset, N, list_I, &L, OUTPUT_filename);

//  cout << endl << "*****************************  Best pairwise model with K interactions:  **********************************" << endl;
//  int Kmax = 40;
//  set<Operator> m1_ranked = PairOp_m1_data_rank(Nset, N);
//  Heuristic_Model_K(Nset, N, list_I, m1_ranked, OUTPUT_filename, communities, final_run, Kmax);


  return 0;
}

