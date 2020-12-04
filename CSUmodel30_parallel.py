import dataclasses as dc
import random
import numpy as np
import time
import pandas
from pathlib import Path
import pandas as pd
from matplotlib import cm
import ray
import pdb

from typing import Any

# ### This part has been added by Sadaf for loading the matrices

import tarfile
from math import floor
# all_input is the directory in which my input_npy.tar.gz lives in. Change this accordingly based on your directories
DATA_DIR = 'data/'
TAR_FILENAME = 'inputs.tar.gz'
DATA_FILENAME = 'input.npy'

full_tar_path = str(Path(__file__).parent / (DATA_DIR + TAR_FILENAME))
my_tar = tarfile.open(full_tar_path)  # Please download input_npy.tar.gz
npy_path = str(Path(__file__).parent / DATA_DIR)
my_tar.extractall(path=npy_path)  # specify which folder to extract to
full_npy_path = npy_path + '/' + DATA_FILENAME
count_matrix = np.load(full_npy_path)  # loading
#count_matrix.shape
T = count_matrix
num_loc = np.shape(T)[-1]



ray.init()
OutVerbosity = 0
Tmat = 1#0 for random, 1 means read actual transition matrix from file
Ndays = 80
Qvar= 2
num_sim_iters = Ndays*24
num_agents = 10000
InitialInfections = 5
alpha = 1
R0=2.5
gamma=0.33
beta = R0*gamma
P = 0.5
pv = [0.5, 0.4, 0.09, 0.01]#probability one becomes asymptomatic, mild, severe, critical
infection_risk_factor = 1
mu = 0.1 # coefficient for wastewater risk
days_to_close_wastewater = 0

print('Number of days', Ndays)




print("Case distribution", pv[1])
#time.sleep(3)
vec0 = np.arange(num_sim_iters+1)
vec = np.zeros(num_sim_iters+1)
vec1 = np.ones(num_sim_iters+1)
vecDays = np.arange(Ndays+1)
vecNewInfections = np.zeros(Ndays+1)
vecNewIso = np.zeros(Ndays+1)
vecQuarantine = np.zeros(num_sim_iters+1)

HealthStatus = np.zeros((num_sim_iters, num_agents))
LocationStatus = np.zeros((num_sim_iters, num_agents))

#Locations1 = ["Weber Building", "Computer Science Building", "Statistics Building", "Parmalee", "Braiden", "Green Hall", "LSC", "Gifford", "Aggie Village", "Off-Campus"]
Locations = ['Academic Village Aspen Hall', 'R', 163,'Academic Village Commons', 'U', 33,  'Academic Village Engineering', 'R', 214, 'Academic Village Honors','R', 146, 
             'Administration', 'A', 33, 'Alder Hall', 'U', 33, 'Allison Hall', 'R', 332, 'Alpha Gamma Rho', 'G', 33,  'Alpine Hall', 'U', 33, 'Ammons Hall', 'U', 33, 
             'Anderson Academic Center', 'U', 33, 'Animal Sciences', 'U', 33, 'Aylesworth Hall', 'U', 33, 'Behavioral Sciences', 'U', 33,  'Biology/Research', 'U', 33, 
             'Braiden Hall', 'R', 505, 'CSU Health and Medical Center', 'U', 33, 'Canvas Stadium', 'U', 33, 'Centennial Hall', 'U', 33, 
             'Center for Environmental Management of Military Lands', 'U', 33, 'Center for Literary Publishing', 'U', 33, 'Central Receiving', 'U', 33, 
             'Chemistry', 'D', 33, 'Chemistry/Research', 'D', 33, 'Chi Omega','G', 33, 'Clark', 'U', 33, 'College of Veterinary Medicine and Biomedical Sciences', 'U', 33, 
             'Computer Science', 'D', 33, 'Confucius Institute', 'U', 33,  'Corbett Hall', 'R', 207,  'Danforth Chapel', 'U', 33 ,'Delta Delta Delta', 'G', 33, 
             'Design Building', 'U', 33, 'Durrell Center','U', 33, 'Durward Hall', 'R', 360, 'Eddy Hall','U', 33, 'Education','U', 33, 'Edwards Hall', 'R', 349, 
             'Engineering', 'U', 33, 'Environmental Health', 'U', 33, 'Facilities Services Center North', 'F', 33, 'Facilities Services Center South', 'F', 33, 
             'Facilities South', 'F', 33, 'FarmHouse Fraternity', 'U', 33, 'Farmhouse','U', 33, 'Forestry', 'D', 33, 'Gamma Phi Beta', 'G', 33, 'General Services', 'U', 33,  
             'Gibbons', 'U', 33, 'Gifford', 'U', 33, 'Glover', 'U', 33, 'Green Hall', 'U', 33, 'Guggenheim Hall', 'U', 33, 'Hartshorn Health Center', 'U', 33, 
             'Health and Exercise Science', 'D', 33, 'Heating Plant', 'U', 33, 'Indoor Practice Facility', 'U', 33, 'Industrial Sciences Lab', 'U', 33, 
             'Ingersoll Hall', 'R', 335, 'Insectary','U', 33,  'Johnson Hall', 'U', 33, 'Kappa Alpha Theta', 'G', 33, 'Kappa Delta', 'G', 33, 'Kappa Kappa Gamma', 'G', 33, 
             'Kappa Sigma', 'U', 33, 'Lake Street Parking Garage', 'U', 33, 'Laurel Hall', 'R', 273, 'Library Depository', 'U', 33, 'Lory Student Center', 'U', 33, 
             'McGraw Athletic Center', 'U', 33, 'Microbiology', 'D', 33,  'Microwave Tower Building', 'U', 33, 'Military Annex', 'U', 33,  'Military Science','U', 33,
             'Moby Arena', 'U', 33, 'Moby Building','U', 33,  'Moby Pool', 'U', 33, 'Molecular and Radiological Biosciences', 'U', 33, 'Morgan Library', 'U', 33, 
             'National Center for Genetic Resources Preservation','U', 33,  'Natural Resources', 'U', 33, 'Natural Resources Research Lab', 'U', 33, 
             'Natural and Environmental Sciences', 'U', 33, 'Newsom Hall', 'R', 301, 'ON-CAMPUS', 'U', 33, 'Occupational Therapy', 'U', 33, 
             'Occupational Therapy Annex', 'U', 33, 'Painter Center', 'U', 33, 'Palmer Center', 'U', 33, 'Parmelee Hall', 'R', 419, 'Pathology', 'U', 33, 
             'Paul B. Thayer TRIO House','U', 33,  'Pavilion', 'U', 33, 'Phi Delta Theta', 'U', 33,  'Physiology', 'D', 33, 'Pi Beta Phi', 'G', 33, 
             'Pi Kappa Phi', 'U', 33,'Pinon Hall','U', 33,  'Plant Growth Facilities', 'U', 33, 'Plant Science', 'U', 33, 'Preconstruction Center', 'U', 33, 
             'Rockwell Hall', 'U', 33, 'Rockwell Hall West', 'U', 33,'Routt Hall', 'U', 33,'Sage Hall', 'U', 33,'Shepardson','U', 33, 'Sigma Nu', 'U', 33,
             'Sigma Phi Epsilon', 'U', 33,'Sigma Pi','U', 33, 'South College Gymnasium', 'U', 33,'Spruce Hall', 'U', 33,'Statistics Building', 'U', 33,
             'Student Recreation Center', 'U', 33,'Student Services', 'U', 33,'Study Cube','U', 33, 'Summit Hall', 'R', 395, 'Surplus Store', 'U', 33,
             'TILT Building', 'U', 33,'UCA East','U', 33, 'UNKNOWN', 'U', 5000,'University Center for the Arts','U', 33, 'University Square','U', 33, 
             'University Village Center', 'R', 462,'Visual Arts','U', 33, 'Wagar', 'U', 33,'Walnut','U', 33, 'Weber', 'U', 33,'Weed Research Lab', 'U', 33,
             'Westfall Hall', 'R', 345,'Yates Hall','U', 33, 'Zeta Tau Alpha','U', 33]



num_loc = int(len(Locations)/3)
print("Number of locations", num_loc)

# build location dictionary such that location_dictionary has keys as location_names
# names and values as a 3-tuple (index, location_label, capacity)
location_names = [Locations[3*i] for i in range(num_loc)]
location_dictionary = {loc_name:(loc_idx,Locations[loc_idx*3+1],Locations[loc_idx*3+2]) for loc_idx,loc_name in enumerate(location_names)}
LocationStatus = np.zeros((num_sim_iters, num_agents))

locationDictionaryLocal = ray.put(location_dictionary)
# import wastewater dataframe
WW_FILENAME = 'waste_water.csv'
ww_fullpath =  str(Path(__file__).parent / (DATA_DIR + WW_FILENAME))
ww_df = pd.read_csv(ww_fullpath)
monitored_dorms = ['Braiden Hall','Allison Hall', 'Summit Hall']



RiskStatus = np.zeros((num_sim_iters+1, num_loc))
BuildingPop = np.zeros((num_sim_iters+1, num_loc))
#3num_res = len(Residences)

if OutVerbosity > 0:
#    print("Number of Residences", num_res)
    print("Number of locations", num_loc)




#define transition matrix
##T = np.random.rand(num_loc, num_loc)
##ss = np.zeros(num_loc)
##for i in range(num_loc):#make it more likely that you don't move
##    T[i,i] = T[i,i]+ 4.0
##ss = np.sum(T, axis = 0)
##
##for j in range(num_loc):
##    for i in range(num_loc):
##        T[i,j] = T[i,j]/ss[j]
##
##ss = np.sum(T, axis = 0)
##if OutVerbosity > 0:


#initialize the transition matrix
if Tmat == 0:
    #T = np.random.rand(num_loc, num_loc)
    T = np.random.rand(7, 24, num_loc, num_loc)




from dataclasses import field
from typing import List
@dc.dataclass
class LocationX:
    LocationID: int
    LocationName: str = 'Unknown'
    LocationCapacity: str = 'Uknown'
    LocationPurpose: str = 'Unknown'#residence, department, administration, ...
    risk_value: float = 0
    waste_water: float = 0
    pop: int = 0
    agentlist: List[int] = field(default_factory=list)


@dc.dataclass
class AgentX():
    uniqueID: int
    age: int
    nloc: int = 0#location index
    immune: str = 'False'
    infected: str = 'False'#True/False
    symptoms: str = 'False'#asymptomatic, mild, severe, critical
    shedding: str = 'False'
    isolated: str = 'False'
    days_sick: int = 0
    days_isolated: int = 0
    role: str = 'Unknown'
    residence: str = 'UNKNOWN'

#######################################################################


#########################################################################
def move_agent(agent, day, hour, num_loc,TransitionMat,locationDictionaryLocal):
    #define sample transition matrix
#    print("in move agent", day, hour)
#    time.sleep(.11)
    A= TransitionMat[day,hour,:,:]#columns sum to one
    vecp = np.zeros(num_loc)
    vecq = np.zeros(num_loc)
    l = np.zeros(num_loc)

    #this is the agent location vector

#    l = numpy.array([ 1, 0 ,0, 0, 0, 0, 0, 0, 0, 0])
    index = agent.nloc
    l[index] = 1.0#location is the standard basis vector with a one in the agents location
    d = A@l

    # agent has a "high" probability of going home
    p_gohome = 0.8
    goHome = random.random()
    if goHome < p_gohome:
        home_idx = locationDictionaryLocal[agent.residence][0]
        vecq[home_idx] = 1
    else:

        #now decide where the agent has moved using a PDF
        vecp[0] = d[0]#this is the cdf
        x = random.random()
        found = 'False'
        for i in range(1,num_loc):
            vecp[i] = d[i]+vecp[i-1]
            if x < vecp[i-1]:
                if i == 85:
                    k = random.randint(0,num_loc-1)
                    vecq[k]=1
                #elif locs3[k] == 'R' ADAPT CODE TO SEND PEOPLE HOME
                #    home = agent.residence
                #    print('Need to return HOME',)
                #    k = agent()
                else:
                    vecq[i-1] = 1
                found = 'True'
                break
        if found == 'False':
            vecq[num_loc-1]=1.0

    #print("random", x)
    #print("The cdf", vecp)
    for i in range(num_loc):
        if vecq[i] == 1.0:
            agent.nloc = i
    return agent
  
@ray.remote
def mega_work(agents,day, hour, num_loc,TransitionMat,locationDictionaryLocal):
    result_list = []
    for agent in agents:
        if agent.isolated == 'False':
            result_list.append(move_agent(agent,day, hour, num_loc,TransitionMat,locationDictionaryLocal))      
        else:
            result_list.append(agent)
    return result_list
########################################################3
#######################################################################
#init locations
locs3 = [LocationX(n) for n in range(num_loc)]

#if OutVerbosity > 0:
for i in range(num_loc): #Attach the names to the building
    locs3[i].LocationName = Locations[3*i]
    locs3[i].LocationPurpose = Locations[3*i+1]
    locs3[i].LocationCapacity = Locations[3*i+2]
    #print(locs3[i])

print('Normalize Transition Matrix')
for m in range(7):
    for n in range(24):
        C = T[m,n,:,:]
        #print(C.shape)
        #time.sleep(10)
        ss = np.zeros(num_loc)
        ss = np.sum(C, axis = 0)
        for j in range(num_loc):
            for i in range(num_loc):
                #print("Entry",i,j,C[i,j])
                if ss[j] != 0:
                    C[i,j] = C[i,j]/ss[j]
        T[m,n,:,:] = C#replace with norm transition matrix

#print("Orig Sum of items with axis = 0 : ", ss)
for m in range(7):
    for n in range(24):
        for k in range(num_loc):
            if T[m,n,k,k] == 0:
                if locs3[k].LocationPurpose == 'R':
                    T[m,n,k,k] = 1
                else:
                    T[m,n,k,k] = 0.2

for m in range(7):
    for n in range(24):
        C = T[m,n,:,:]
        #print(C.shape)
        #time.sleep(10)
        ss = np.zeros(num_loc)
        ss = np.sum(C, axis = 0)
        for j in range(num_loc):
            for i in range(num_loc):
                #print("Entry",i,j,C[i,j])
                if ss[j] != 0:
                    C[i,j] = C[i,j]/ss[j]
        T[m,n,:,:] = C#replace with norm transition matrix
print('Done')

TransitionMatLocal = ray.put(T)

#initialize the agents
agents3 = [AgentX(n, random.randint(17,70)) for n in range(num_agents)]
#Place all agents in initial locations and infect subpopulation
i = 0;#number of agents that have found beds
#Fill the residence halls and then add random locations
#if location is a residence hall and below capacity then add agent here.
for k in range(num_loc):#find a bed in location k?
    for j in range(locs3[k].LocationCapacity):
        if locs3[k].LocationPurpose == 'R':
            while i < num_agents and locs3[k].pop < locs3[k].LocationCapacity:
                #print("Adding agent",i,"to residence",k)
                agents3[i].nloc = k
                agents3[i].residence = locs3[k].LocationName
                locs3[k].pop = locs3[k].pop + 1
                locs3[k].agentlist.append(i)
                i = i+1

#print("tricky spot", i)
#time.sleep(30)
#now complete by adding agents to random locations
for j in range(i,num_agents):
    k = random.randint(0,num_loc-1)
    agents3[j].nloc = k#put the agent in the initial location nloc (int)
    locs3[k].agentlist.append(j)#put agent i in its location

for k in range(num_loc):#save population
    #print("Finding populations", k , locs3[k])
    locs3[k].pop = len(locs3[k].agentlist)
    BuildingPop[0,k] = locs3[k].pop
    #time.sleep(2)


#introduce initial infections in one location (Braiden this time)
iii_name = 'Braiden Hall'
iii=location_dictionary[iii_name][0]#index of location where individuals are initially infected
for iii_agent in locs3[iii].agentlist:
    if locs3[iii].risk_value < InitialInfections:
        agents3[iii_agent].infected = 'True'#seed infections.
        agents3[iii_agent].shedding = 'True'
        locs3[iii].risk_value = locs3[iii].risk_value+1
    else:
        break
        #print("MAED OY", locs3[iii])
        #time.sleep(1)

RiskStatus[0,iii] = locs3[iii].risk_value
vec[0] = InitialInfections/num_agents
vecNewInfections[0] = InitialInfections
vecNewIso[0] = 0
vecQuarantine[0] = 0
#print(InitialInfections, num_agents, vec[0])
#time.sleep(2)

#show initial building populations and agent propoerties
if OutVerbosity > 0:
    for m in range(num_loc): #loop over the location
        print(locs3[m])
    for i in range(num_agents):
        print(agents3[i])
###############################################################################
###START OF SIMULATION
new_infections = 0
new_iso = 0
# parallel parameter
batch_size = 200
batch_num = int(num_agents/batch_size)
for i in range(num_sim_iters):#iteration loop (i iters (units are hrs))
    t0 = time.time()
    hour = i%24#hour of day in military time
    days = floor(i/24)#total number of days that have elapsed
    day = days%7#day of the week 0-6
    print('Time: hour, day, days, iter/hours',hour, day, days, i)

    if OutVerbosity > -1:
        print("Outer loop", i)

    #change agents location according to transition matrix
    agents3_result = []
    for j in range(batch_num):#loop over all agents
        agents_list = [agents3[k] for k in np.arange(j*batch_size,(j+1)*batch_size)]
        agents3_result.append(mega_work.remote(agents_list,day, hour, num_loc,TransitionMatLocal,locationDictionaryLocal))
        # if not isolated, change agents location according to transition matrix
        #if agents3[j].isolated == 'False':
        #    agents3[j] = move_agent.remote(agents3[j], day, hour, num_loc,TransitionMatLocal,locationDictionaryLocal) #this updates agent index location nloc
    agents3 = [item for batch_item in ray.get(agents3_result) for item in batch_item ] 
    for loc in locs3:
        loc.agentlist = []
    for j in range(num_agents):
        if agents3[j].isolated=='False':
            locs3[agents3[j].nloc].agentlist.append(j)
        if OutVerbosity > 0:
            print("New Loc / agent",locs3[agents3[j].nloc],j)
        #update risk status of building after agents have moved
        if agents3[j].infected == 'True' and agents3[j].isolated == 'False':
            loc = agents3[j].nloc
            RiskStatus[i+1,loc] = RiskStatus[i+1,loc]+1
    #update building population and risk_value
    for m in range(num_loc): #loop over the locations
        locs3[m].pop = len(locs3[m].agentlist) #number of agents in this loc
        locs3[m].risk_value = RiskStatus[i+1,m]
        BuildingPop[i+1,m] = locs3[m].pop

    #new round of infections
    for j in range(num_agents):
        if agents3[j].infected == 'False' and agents3[j].immune == 'False':#you can get it
           loc = agents3[j].nloc#current location of agent
           if locs3[loc].LocationPurpose == 'R':
               risk  = locs3[loc].risk_value/locs3[loc].LocationCapacity#the risk value for this agents location
           else:
               risk = 0 # we might want to change this value to some general value
           Frisk=0.01
           Brisk = 0.025
           # if it's one of the monitored dorms(wastewater) add wastewater to it
           if locs3[loc].LocationName in monitored_dorms and days < days_to_close_wastewater:
               ww_risk = ww_df[locs3[loc].LocationName].iloc[days]/location_dictionary[locs3[loc].LocationName][2]
               ww_risk = min(0.8,ww_risk)
           else:
               ww_risk = 0
           xx = (Frisk+ min([risk,1])*Brisk + mu*ww_risk)*infection_risk_factor
           #xx = 0.005+min([risk*0.001,1])#risk value is number of shedders in location loc
           #xx = alpha*np.random.poisson(beta*risk)
           #print('infection risk',i,j,xx)
           #print("Location",loc, locs3[loc].LocationName, "Risk", risk, "Probabity of infection",xx)
           #time.sleep(.2)
           #if (xx>np.random.uniform(.01,.9)):#you are sick now
           if (xx>np.random.uniform(Frisk*infection_risk_factor,1)):
               agents3[j].infected = 'True'#this subject is infected now
               new_infections += 1
               y = random.randint(0,9)
               if (y>5) :
                   agents3[j].symptoms = 'True'#half of infecteds are symptomatic
               z = random.randint(0,9)
               if (z>Qvar) :
                   new_iso += 1
                   agents3[j].isolated = 'True'#some fraction of positives are isolated
                   #need to move to isolation location
                   #move them to a parallel universe now
                   locs3[agents3[j].nloc].agentlist.remove(j)
                   agents3[j].nloc = None
                   #agents3[j].infected = 'True'
                   #agents3[j].immune = 'True'
                   agents3[j].symptoms = 'False'
    
    
    # force 1-3 agents living off-camputs(UNKNOWN) get infected at the end of weekend, assume we start from Monday
    if i>0 and i%(7*24) == 0:
        if days<50:
            num_weekend_infected = np.random.randint(1,4)
        else:
            num_weekend_infected = np.random.randint(3,5)
        weekend_count = 0
        j = 0
        while(weekend_count<num_weekend_infected):
            # find a suceptible agent living off-campus
            if agents3[j].infected == 'False' and agents3[j].immune == 'False' and agents3[j].residence == 'UNKNOWN':
                agents3[j].infected = 'True'
                new_infections += 1
                weekend_count += 1
            j += 1
            if j >= num_agents-1:
                print('No more off-campus agents can be infected')
                break
    
    # compute daily new cases
    if i>0 and (i+1)%24==0:
        vecNewInfections[days+1] =  new_infections
        print('Daily new infections: %d'% new_infections)
        vecNewIso[days+1] = new_iso
        new_infections = 0
        new_iso = 0

    #count the number of infections
    num_infected = 0#these are just initializations, number will be computed
    frac_infected = 0#just resetting running sums
    num_isolated = 0
    frac_isolated = 0
    for j in range(num_agents):#loop over all agents
        #if OutVerbosity > 0:
        #    print("agent index and location:",j, agents3[j].nloc)
        if agents3[j].infected == 'True' and agents3[j].isolated == 'False':
            agents3[j].days_sick = agents3[j].days_sick + 1
            if agents3[j].days_sick > np.random.poisson(14*24):#recovered?
                agents3[j].immune = 'True'
                agents3[j].symptoms = 'False'
                agents3[j].infected = 'False'
                #HealthStatus[i,j] = 0#reset health status to healthy
                agents3[j].isolated = 'False'#remove from isolation!
                #agents3[j].nloc = location_dictionary[agents3[j].residence][0]# now start from your residence
                #loc3[agents3[j].nloc].agentlist.append(j) 
            else: #agent is sick
                HealthStatus[i,j] = 1
                num_infected = num_infected + 1
                frac_infected = float(num_infected)/float(num_agents)
        elif agents3[j].infected == 'False':#if you aren't sick you are healthy!
            HealthStatus[i,j] = 0
        
        if agents3[j].isolated == 'True':
            agents3[j].days_isolated = agents3[j].days_isolated + 1
            if agents3[j].days_isolated > np.random.poisson(14*24):#recovered?
                agents3[j].isolated = 'False'
                agents3[j].immune = 'True'
                agents3[j].symptoms = 'False'
                agents3[j].infected = 'False'
                agents3[j].nloc = location_dictionary[agents3[j].residence][0]# now start from your residence
                locs3[agents3[j].nloc].agentlist.append(j)
            else:
                HealthStatus[i,j] = 1
                num_isolated += 1
                frac_isolated = float(num_isolated)/float(num_agents)
        LocationStatus[i,j] = agents3[j].nloc#location of agent j at time i

    #vec[i+1] = frac_infected
    vec[i+1] = frac_infected + frac_isolated
    vecQuarantine[i+1] = frac_isolated
    print("Fraction infected %.5f"% frac_infected)
    print("Fraction isolated %.5f"% frac_isolated)
    print("Fraction total %.5f"% (frac_infected + frac_isolated))
    t1 = time.time()
    print('This iteration cost(seconds)', (t1-t0))



###End of imulation iterations loop
##############################################################################
##############################################################################


import matplotlib.pyplot as plt

plt.figure()
plt.scatter(np.array(vec0), vec,color='red')
plt.xlabel('Simulation iteration')
plt.ylabel('Fraction infected')
aa = 'Number of Initial Infections '+str(InitialInfections)
plt.title(aa)
aa = 'FracInfectionsPlot'+'B'+str(iii)+'II'+str(InitialInfections)+'Brisk'+str(Brisk)+'Q'+str(Qvar)+'.png'
plt.savefig(aa)
#plt.show()

# plot daily new cases
fig,ax = plt.subplots(1)
fig.suptitle('New cases(infections/quarantines) per day')
ax.bar(np.array(vecDays), vecNewInfections,color='r',width=0.25,label='New Infections')
ax.bar(np.array(vecDays)+0.25, vecNewIso, color='b',width=0.25,label='New Quarantines')
ax.legend()
ax.set_xlabel('Days')
ax.set_ylabel('new cases')
fig.tight_layout()
new_case_figname = 'DailyNewInfectionsPlusQuarantines'+'B'+str(iii)+'II'+str(InitialInfections)+'Brisk'+str(Brisk)+'Q'+str(Qvar)+'.png'
fig.savefig(new_case_figname)


# plot risk status vs iterations
fig_size = (8.5, 5)
plt.figure(figsize = fig_size)
residence_risk_list = [0,6,15,58,89,115,128]
risk_status_color = cm.get_cmap('tab20',len(residence_risk_list))
risk_figname ='BuildingRistStatus'+'B'+str(iii)+'II'+str(InitialInfections)+'Brisk'+str(Brisk)+'Q'+str(Qvar)+'.png'

for idx, rsk_idx in enumerate(residence_risk_list):
    plt.plot(vec0/24, RiskStatus[:,rsk_idx:(rsk_idx+1)], color=risk_status_color.colors[idx],label=locs3[rsk_idx].LocationName)
plt.legend(bbox_to_anchor=(1.05,1),loc='upper left')
plt.xlabel('Days')
plt.ylabel('Risk Status of Building')
plt.tight_layout()
plt.savefig(risk_figname)
#plt.plot(vec1*locs3[0].LocationCapacity, color = 'red')
#plt.plot(vec0, RiskStatus[:,0:1],color="red",label = locs3[0].LocationName)
#plt.plot(vec1*locs3[2].LocationCapacity, color = 'black')
#plt.plot(vec0, RiskStatus[:,2:3],color="black",label = locs3[2].LocationName)
#plt.plot(vec0, RiskStatus[:,89:90],color="yellow",label = locs3[89].LocationName)
#plt.plot(vec0, RiskStatus[:,83:84],color="cyan",label = locs3[83].LocationName)
#plt.plot(vec0, RiskStatus[:,96:97],color="blue",label = locs3[96].LocationName)
#plt.plot(vec1*locs3[115].LocationCapacity, color = 'yellow')
#plt.plot(vec0, RiskStatus[:,115:116],color="orange",label = locs3[115].LocationName)
#plt.plot(vec1*locs3[119].LocationCapacity, color = 'cyan')
#plt.scatter(vec0, RiskStatus[:,119:120],color="cyan",label = locs3[119].LocationName)
#plt.plot(vec1*locs3[58].LocationCapacity, color = 'green')
#plt.plot(vec0, RiskStatus[:,58:59],color="green",label = locs3[58].LocationName)
#plt.plot(vec1*locs3[128].LocationCapacity, color = 'brown')
#plt.plot(vec0, RiskStatus[:,128:129],color="brown",label = locs3[128].LocationName)
#plt.plot(vec1*locs3[15].LocationCapacity, color = 'purple')
#plt.plot(vec0, RiskStatus[:,15:16],color="purple",label = locs3[15].LocationName)
#plt.legend()
#plt.xlabel('Simulation iteration')
#plt.ylabel('Risk Status of Building')

#plt.figure()
#plt.plot(vec1*locs3[1].LocationCapacity, color = 'magenta')
#plt.plot(vec0, RiskStatus[:,1:2],color="magenta",label = locs3[1].LocationName)
#plt.plot(vec1*locs3[112].LocationCapacity, color = 'purple')
#plt.plot(vec0, RiskStatus[:,112:113],color="cyan",label = locs3[112].LocationName)
#plt.plot(vec1*locs3[126].LocationCapacity, color = 'blue')
#plt.plot(vec0, RiskStatus[:,126:127],color="blue",label = locs3[126].LocationName)
#plt.plot(vec0, RiskStatus[:,110:111],color="red",label = locs3[110].LocationName)
#plt.plot(vec0, RiskStatus[:,105:106],color="magenta",label = locs3[105].LocationName)
#plt.plot(vec0, RiskStatus[:,97:98],color="purple",label = locs3[97].LocationName)
#plt.plot(vec0, RiskStatus[:,78:79],color="black",label = locs3[78].LocationName)
#plt.plot(vec0, RiskStatus[:,75:76],color="green",label = locs3[75].LocationName)
#plt.plot(vec0, RiskStatus[:,68:69],color="brown",label = locs3[68].LocationName)
#plt.legend()
#plt.xlabel('Simulation iteration')
#plt.ylabel('Risk Status of Building')


# plot building population vs. simulation iterations
plt.figure()
fig_size = (8.5, 5)
plt.figure(figsize = fig_size)
building_pop_list = [0,6,15,58,89,115,128]
building_pop_color = cm.get_cmap('tab20',len(residence_risk_list))
pop_figname = 'BuildingPopulation'+'B'+str(iii)+'II'+str(InitialInfections)+'Brisk'+str(Brisk)+'Q'+str(Qvar)+'.png'
for idx, pop_idx in enumerate(building_pop_list):
    plt.plot(vec0/24, BuildingPop[:,pop_idx:(pop_idx+1)], color=risk_status_color.colors[idx],label=locs3[pop_idx].LocationName)
plt.legend(bbox_to_anchor=(1.02,1),loc='upper left')
plt.xlabel('Days')
plt.ylabel('Building Population')
plt.tight_layout()
plt.savefig(pop_figname)


#plt.plot(vec1*locs3[3].LocationCapacity, color = 'red')
#plt.plot(vec0, BuildingPop[:,3:4],color="red",label = locs3[3].LocationName)
#plt.plot(vec1*locs3[0].LocationCapacity, color = 'blue')
#plt.plot(vec0, BuildingPop[:,0:1],color="blue",label = locs3[0].LocationName)
#plt.plot(vec1*locs3[2].LocationCapacity, color = 'orange')
#plt.plot(vec0, BuildingPop[:,2:3],color="orange",label = locs3[2].LocationName)
#plt.plot(vec0, BuildingPop[:,89:90],color="yellow",label = locs3[89].LocationName)
#plt.plot(vec0, BuildingPop[:,83:84],color="cyan",label = locs3[83].LocationName)
#plt.plot(vec0, BuildingPop[:,96:97],color="black",label = locs3[96].LocationName)
#plt.plot(vec1*locs3[115].LocationCapacity, color = 'yellow')
#plt.scatter(vec0, BuildingPop[:,115:116],color="yellow",label = locs3[115].LocationName)
#plt.plot(vec1*locs3[119].LocationCapacity, color = 'cyan')
#plt.scatter(vec0, RiskStatus[:,119:120],color="cyan",label = locs3[119].LocationName)
#plt.plot(vec1*locs3[58].LocationCapacity, color = 'green')
#plt.plot(vec0, BuildingPop[:,58:59],color="green",label = locs3[58].LocationName)
#plt.plot(vec1*locs3[128].LocationCapacity, color = 'brown')
#plt.plot(vec0, BuildingPop[:,128:129],color="brown",label = locs3[128].LocationName)
#plt.plot(vec1*locs3[15].LocationCapacity, color = 'purple')
#plt.plot(vec0, BuildingPop[:,15:16],color="purple",label = locs3[15].LocationName)
#plt.legend()
#plt.xlabel('Simulation iteration')
#plt.ylabel('Building Population')

#plt.figure()
#plt.plot(vec1*locs3[1].LocationCapacity, color = 'magenta')
#plt.plot(vec0, BuildingPop[:,1:2],color="orange",label = locs3[1].LocationName)
#plt.plot(vec1*locs3[112].LocationCapacity, color = 'purple')
#plt.plot(vec0, BuildingPop[:,112:113],color="purple",label = locs3[112].LocationName)
#plt.plot(vec1*locs3[126].LocationCapacity, color = 'blue')
#plt.plot(vec0, BuildingPop[:,126:127],color="blue",label = locs3[126].LocationName)
#plt.plot(vec0, BuildingPop[:,110:111],color="red",label = locs3[110].LocationName)
#plt.scatter(vec0, BuildingPop[:,105:106],color="yellow",label = locs3[105].LocationName)
#plt.scatter(vec0, BuildingPop[:,97:98],color="purple",label = locs3[97].LocationName)
#plt.plot(vec0, BuildingPop[:,78:79],color="yellow",label = locs3[78].LocationName)
#plt.plot(vec0, BuildingPop[:,75:76],color="green",label = locs3[75].LocationName)
#plt.plot(vec0, BuildingPop[:,68:69],color="brown",label = locs3[68].LocationName)
#plt.legend()
#plt.xlabel('Simulation iteration')
#plt.ylabel('Building Population')




#ax1 = plt.subplot(511)
#plt.plot(vec0, BuildingPop[:,0:1])
#ax1.set_xlabel('Time (hours)')
#plt.ylabel('Population')


#ax2 = plt.subplot(512)
#plt.plot(vec0, BuildingPop[:,2:3])

#ax3 = plt.subplot(513)
#plt.plot(vec0, BuildingPop[:,89:90])
#ax3.set_title(locs3[89].LocationName)

#ax4 = plt.subplot(514)
#plt.plot(vec0, BuildingPop[:,83:84])
#ax4.set_title(locs3[83].LocationName)

#ax5 = plt.subplot(515)
#plt.plot(vec0, BuildingPop[:,96:97])
#ax5.set_title(locs3[96].LocationName)

#from array import *

#T = [[.1, .2, 0, .5, .2], [0, .3, 0, .5, .2], [.2, .5, 0, .3, .0], [.1, .5, 0, .2, .2],[.1, .2, 0, .5, .2]]

#print(T)
#subroutine applies a transition matrix to determine the new location
#of an agent.
#Input: agent at current location
#Output: agent at new location


#agents3[1] = move_agent(agents3[1])

print("Health Status", HealthStatus)
print("Location Status", LocationStatus)
#plt.show()
print("Building Population", BuildingPop)
#plt.show()
print("Risk Status", RiskStatus)
X = pandas.DataFrame(HealthStatus)
aa = 'HealthStatus'+'B'+str(iii)+'II'+str(InitialInfections)+'Brisk'+str(Brisk)+'Q'+str(Qvar)+'.csv'
X.to_csv(aa)
Y = pandas.DataFrame(LocationStatus)
aa = 'LocationStatus'+'B'+str(iii)+'II'+str(InitialInfections)+'Brisk'+str(Brisk)+'Q'+str(Qvar)+'.csv'
Y.to_csv(aa)
Y = pandas.DataFrame(BuildingPop)
aa = 'BuildingPop'+'B'+str(iii)+'II'+str(InitialInfections)+'Brisk'+str(Brisk)+'Q'+str(Qvar)+'.csv'
Y.to_csv(aa)
aa = 'RiskStatus'+'B'+str(iii)+'II'+str(InitialInfections)+'Brisk'+str(Brisk)+'Q'+str(Qvar)+'.csv'
Y = pandas.DataFrame(RiskStatus)
Y.to_csv(aa)

plt.figure()

plt.imshow(RiskStatus)
plt.colorbar()
plt.ylabel('Simulation iteration')
plt.xlabel('Location')
plt.title('Building Risk Value')
#plt.show()
aa = 'RiskStatusPlot'+'B'+str(iii)+'II'+str(InitialInfections)+'Brisk'+str(Brisk)+'Q'+str(Qvar)+'.png'

plt.savefig(aa)

plt.figure()
plt.imshow(HealthStatus)
plt.colorbar()
plt.ylabel('Simulation iteration')
plt.xlabel('Agent Index')
plt.title('Population Health Status')
#plt.show()
aa = 'HealthStatusPlot'+'B'+str(iii)+'II'+str(InitialInfections)+'Brisk'+str(Brisk)+'Q'+str(Qvar)+'.png'
plt.savefig(aa)

plt.figure()
plt.imshow(BuildingPop)
plt.colorbar()
plt.ylabel('Simulation iteration')
plt.xlabel('Location')
plt.title('Building Population')
plt.show()
aa = 'BuildingPopPlot'+'B'+str(iii)+'II'+str(InitialInfections)+'Brisk'+str(Brisk)+'Q'+str(Qvar)+'.png'
plt.savefig(aa)

XX = T[0,0,:,:]
print(XX.shape)
#print(XX(119))
plt.imshow(XX)
plt.colorbar()
plt.ylabel('Location i')
plt.xlabel('Location j')
aa = 'Transition Matrix at Monday 12-1am'
plt.title(aa)
plt.savefig('TransitionM00.png')
YY = T[0,12,:,:]
print(XX.shape)
#print(XX(119))
plt.imshow(YY)
plt.colorbar()
plt.ylabel('Location i')
plt.xlabel('Location j')
aa = 'Transition Matrix at Monday 12-1pm'
plt.title(aa)
plt.savefig('TransitionM12.png')




for i in range(num_loc):
    print(locs3[i])


#ss2 = np.sum(TTT, axis = 0)
#if OutVerbosity > 0:
#print("Transition Matrix", TTT)
#print("Normed Sum of items with axis = 0 : ", ss2)



#print("Fraction infected", vec)
print("locations", Locations[84])
print("Location", locs3[0].LocationName,"Risk Status",RiskStatus[:,0:1])

import scipy.linalg as la
eigvals, eigvecs = la.eig(XX)
print(sorted(abs(eigvals)))


print("Initial Population", BuildingPop[0,:])
