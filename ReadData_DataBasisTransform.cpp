#include <iostream>
#include <fstream>
#include <sstream>
#include <list>
#include <bitset>
#include <map>
#include <cmath>       /* tgamma */

//using namespace std;

/********************************************************************/
/**************************    CONSTANTS    *************************/
/********************************************************************/
#include "data.h"

/******************************************************************************/
/**************************     READ FILE    **********************************/
/******************************************************************************/

/**************************  READ PARTITION  ********************************/



/**************    READ DATA and STORE them in Nset    ************************/
map<uint32_t, unsigned int> read_datafile(unsigned int *N, string filename = datafile)    // O(N)  //N = data set size
{
  string line, line2;     uint32_t nb = 0;
  (*N) = 0;            // N = dataset size (global variable)
  string INPUT_file = INPUT_folder + filename;
  cout << endl << "--->> Read \"" << filename << "\",\t Build Nset...";

// ***** data are store in Nset:  ********************************
  map<uint32_t, unsigned int> Nset; // Nset[mu] = #of time state mu appears in the data set

  ifstream myfile (INPUT_file.c_str());
  if (myfile.is_open())
  {
    while ( getline (myfile,line))
    {
      line2 = line.substr (0,n);          //take the n first characters of line
      nb = bitset<n>(line2).to_ulong();   //convert string line2 into a binary integer
      Nset[nb] += 1;
      //cout << line << endl;   //cout << nb << " :  " << bitset<n>(nb) << endl;
      (*N)++;
    }
    myfile.close();
  }
  else cout << "Unable to open file";

  cout << "\t\t data size N = " << (*N) << endl;

  return Nset;
}

/******************************************************************************/
/*********************     CHANGE of BASIS: one datapoint  ********************/
/******************************************************************************/
// Given a choice of a model (defined by the m basis vector) --> return the new m-state (state in the new m-basis)
// Rem: must have m <= n
uint32_t transform_mu_basis(uint32_t mu, list<uint32_t> basis)
{
  uint32_t un_i = 1;
  uint32_t final_mu = 0;

  list<uint32_t>::iterator phi_i;

  for(phi_i = basis.begin(); phi_i != basis.end(); ++phi_i)
  {
    if ( (bitset<n>( (*phi_i) & mu ).count() % 2) == 1) // odd number of 1, i.e. sig_i = 1
      {
        final_mu += un_i;
      }
    un_i = (un_i << 1);
  }

  return final_mu;
}

/******************************************************************************/
/************************** K_SET *********************************************/
/******************************************************************************/
// Build Kset for the states written in the basis of the m-chosen independent
// operator on which the SC model is based:

map<uint32_t, unsigned int> build_Kset(map<uint32_t, unsigned int> Nset, list<uint32_t> Basis_SCModel, bool print_bool=false)
// sig_m = sig in the new basis and cut on the m first spins
// Kset[sig_m] = #of time state mu_m appears in the data set
{
  map<uint32_t, unsigned int>::iterator it;
  map<uint32_t, unsigned int > Kset;

  uint32_t s;        // initial state
  uint32_t sig_m;    // transformed state and to the m first spins

  unsigned int ks=0; // number of time state s appear in the dataset

  cout << endl << "--->> Build Kset..." << endl;

//Build Kset:
  for (it = Nset.begin(); it!=Nset.end(); ++it)
  {
    s = it->first;       // state s
    ks = it->second;    // # of times s appears in the data set
    sig_m = transform_mu_basis(s, Basis_SCModel);
//    sig_m = bitset<m>(bitset<m>(mu).to_string()).to_ulong(); //bitset<m>(mu).to_ulong(); // mu|m
    if (print_bool)  {  cout << s << ": \t" << bitset<n>(s) << " \t" << sig_m << ": \t" << bitset<n>(sig_m) << endl; }

    Kset[sig_m] += ks;
    //Kset[mu_m].second.push_back(make_pair(mu, N_mu));
  }
  cout << endl;

  return Kset;
}




/******************************************************************************/
/************************** CUT STATE *****************************************/
/******************************************************************************/

uint32_t state_cut(uint32_t state, uint32_t part)
{
  uint32_t new_state = 0;
  int new_position = 0;
  int new_var_position;
  for (int var = 0; var < n; var++)
  {
    new_var_position = 1;

    if (bitset<n>(part)[var] == 1 && bitset<n>(state)[var] == 1)
    {
      new_var_position = new_var_position << new_position;
      new_state = new_state + new_var_position;

    }
    if (bitset<n>(part)[var] == 1)
    {
      new_position++;
    }
  }
  return new_state;
}

/******************************************************************************/
/************************** CUT KSET ******************************************/
/******************************************************************************/

map<uint32_t, unsigned int> build_cut_Kset(map<uint32_t, unsigned int> Kset, uint32_t community)
// sig_m = sig in the new basis and cut on the m first spins
// Kset[sig_m] = #of time state mu_m appears in the data set
{
  map<uint32_t, unsigned int>::iterator it;
  map<uint32_t, unsigned int > Kset_buffer;
  uint32_t new_state;
  uint32_t s;        // initial state


  unsigned int ks=0; // number of time state s appear in the dataset

  cout << endl << "--->> Build cut Kset..." << endl;

//Build cut Kset:
  for (it = Kset.begin(); it!=Kset.end(); ++it)
  {
    s = it->first;       // state s
    ks = it->second;    // # of times s appears in the data set

    new_state = state_cut(s, community);
    Kset_buffer[new_state] += ks;

  }


  return Kset_buffer;
}