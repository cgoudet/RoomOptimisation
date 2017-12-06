# RoomOptimisation

RoomOptimisation is a mixed integer programming project which consist in allocating a set of ressources de different stakeholders in order to improve the global performances.
In practice, this project focuses on the optimal placement of employees within a set of offices in order to increase the global well-being, according to the company policy.

## Problem description

The company happytal is opening a new office and wants to optimise the placement of its employees on the new offices in order to optimise happyness.
The optimisation aims finding the office to allocate to each person to maximise an objective function representing the global happyness.
The mathematical formulation of the problem is discussed below.

This function contains three components : 

- Diversity. In order to increse the link between services, one wants to maximise the number of represented services within each room.

- Happyness. Total amount of agreement between employees preferences and their respective allocation.

- Neigbouroud. Total amount of agreement between employees sharing a room.

- Equality. On aims at minimising the difference of happyness between employees.

The optimisation is limited by some conditions which must be strictly enforced : 
 - A person can only be attributed a single office
 - A single office can only be attributed a single time at most
 - Two tall employees can not be seated face to each other.
 
 More constraints are present in the model. 
 However, these constraints are only structural : they allow to use a linear formulation of non-linear constraints.

## Data 
Two datasets are needed for this project : 

### Office Properties
The new offices represent the ressorces to be allocated between the employees. 
Each office has a set of properties :

- a room identification (roomID), which define a bloc offices. 
Some variables, such as the diversity of represented services, of the problem are considered at the level of a group of offices.  

- an indication of the position of the office within the room (isLeft).
This indication is used to set some constraints of the allocation, namely to avoid having two tall person seated face to face and having their kenn touching

- the floor of the office (etage). 
If present, the floor of the office is included in the identification of the room : two rooms with same floor but different roomID are labelled as different rooms.

- spatial properties (wc, sonnerie, passage, ... ). 
These properties label each office as having of not a given spatial property. 
These properties can either have a positive impact (being near a window is usually enjoyed) or a negative one (being near the AC).
Each of these properties is then attributed a value : either +1 or -1.
The employees preferences must contain the same variables as to perfom a matching between preferences and actual allocation.

### Employees Data
The employees are the stakeholder to whom the ressources will be allocated.
Each employee must have a given set of properties :

- name of the employee (Nom). The name is used for employees identify easily each other.

- the service to which he/she belongs (service). 
This is a categorical variable, usually encoded in terms of service names.

- the indication of the employee size (isTall).
The variable defines wether the employee is considered tall. 
This is later used in order to avoid sitting tall people face to face.

- spatial preferences (wc, sonnerie, passage, ...).
These properties are matching the ones with same name within offices data.
The content of these variables is a grade which define the importance this person attributes to the allocation of a given property.
Only the relative preference of variable is taken into account so the grading can be set between 0 and 10.

- neighbour preferences (weightPersoI, persoI).
Neighbour preferences allows to determine which colleagues a given employee would be more happy with.
The number of neighbour preferences (I) is let to the user.
The definition of the preferences use two variable : the grade of the will to be in the same room as a given person and the name of this person.


## Mathematical formulation

### Independent Variables
This section describe the various variables creating the phase space of the minimisation problems. 
Some variables are not independent and are included in order to formulate the problem in a linear way. 
Nevertheless, they are included in this section as the minimisation algorithm still includes them as free variables.
The description of the roles of such variables is performed in a dedicated section.

Let X the placement vector such that X_ij=1 iff the office j is occupied by person i. 
X is the central component of the optimisation as its value will dictate the value of the objective function in non intuitive ways.

delta_sr = 1 iff at least one employee from service s is allocated to the room r.

Let minHappy and maxHappy the respective minimum and maximum of employees individal happyness. 
The computation of individual happiness is described in dedicated section.

### Dependent variables
This section describes a set of variables used throughout the mathematical formulation of the problem.
Usually, those variables are a function of inputs and/or independent variables.

#### Diversity 
The computation of diversity within an office starts from the count of the number of employees from each service : Delta_sr.

Let P_is=1 iff the employee i belongs to service s.
Let R_jr=1 iff the office j belongs to room r.
P and R are only function of the input data.

Then one has :
Delta_sr = transpose(P)_si X_ij R_jr

The diversity delta is defined such that delta_sr=1 ssi Delta_sr>0. 
Such a relation is not linear and must be mocked by constraints
Considering K a global upper bound of Delta, then the two following constraints allow delta to be fully determined by Delta.

delta_sr <= Delta_sr 
delta_sr >= Delta_sr / K

In practice, K is taken as the total amount of employees.


#### Spatial Happyness
The Spatial Happyness represent the agreement between employees preferences and their office properties.
Each employee has given a grade to spatial property to denote its liking into having an office with such a property.

Let P_ip = grade attributed by employee i to property P.
P is then normalized such that 
sum_p(P_ip)=1

Let R_jp = 1 iff office j has the property p.

Let F_pp = +1 of the property p has a positive effect on the happyness, -1 else. This is a diagonal matrix.

W_ip = P_ip F_pp is the matrix of employees preferences weighted by the effect of the property

A_ip = X_ij R_jp is the matrix representing the properties attributed to each person.

Then W_ip * A_ip (elementwise multiplication) represent the weighted agreement attributed to a person per properties.

Finally H_i = sum_p(A_ip) represents the total happyness of patient i. 


#### Happyness Difference

The minimum of hapyness is not a linear function of the data.
Instead, a new free variable is created and imposed to be smaller than all any employee happyness while trying to maximise it.

Similar construction is done for maximum.

#### Legs 

Let L_l the number of tall people seated in a bloc of facing offices

Let R_jl = 1 iff office j blongs to the bloc of facing offices l

Let P_it = 1 iff person i has size property t

transpose(P)_ti X_ij R_jl is the number of people with size property p in bloc l.

L_l = transpose(P)_1i X_ij R_jl





### Objective function

The objective funtion is a linear sum of various linear terms :

- diversity within offices : sum_sr( delta_sr )

- total happyness : sum_i(H_i)

- spread of happyness : minHappy - maxHappy


### Constraint


Constrains for imposing delta values
for s, r in services, rooms
delta_sr <= Delta_sr 
delta_sr >= Delta_sr / K

Constraints to impose one tall person per bloc of facing office
for l in facingBlocs 
L_l<=1

spreadHappyness
for i in H
minHappy<= H_i
maxHappt>= H_i