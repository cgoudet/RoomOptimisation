import numpy as np
import pandas as pd
import pulp
import unittest
from pandas.util.testing import assert_frame_equal
from PIL import Image, ImageFont, ImageDraw


class Constraint() :
    """
    The Constraint class proposes a common framework to include various types of constraints into an optimisation model.
    The possible constraints, as well as their usage, are described in the following with examples from the attribution of offices (ressources) allocated to workers (users).
    
    # General usage of constraint 
    
    ## Constraint attributes
    
    All Constraint usage have the same user interface, whatever the type of constraints to inject in the model.
    A Constraint object is created through a single constructor with :
        - mandatory variables : 
            - type : type of constraint (the authorised types are described below)
            - label : the name of the constraint, which is related to the columns of the inputs datasets which contain the data.
        
        - optional variables : 
            - maxWeights : wether to use this constraint in the objective function. 
            - bound : wether to add a constraint on the model. 
            boud = 0 adds no constraint, bound=1 add an upper limit et bound=-1 adds a lower limit.
            - valBound : value to use as a limit for the bound option
            - roomTag : columns indices to group ressources into blocs with shared properties.
            - multi : wether to add users preferences from multiple columns
            - removeSelf : wether to remove the contribution of a user on itself in the objective function and/or constraint.
        
    ## Constraint usage
    The interface of Constraint to include constraints into an optimisation model is common for all types.
    
    - Instantiate your object using the common constructor, includig all your desired option.
    - Call DefineConstraint() with your data to setup your object. 
    This function fills two arrays, wish and dispo, which respectively represent the distribution of weights and properties allocation as a function of the allocation matrix.
    - Call GetObjVal() to include the objective function into the model. 
    If maxWeights has been initialized as false, GetObjeVal will return 0.
    - Call SetConstraint() to include lower or upper limits to your variables.
    SetConstraint() is mandatory for ppBin and ppCat types. 
    In other types, it has no effects if bound=0
    - Once the model is optimize, a check on the distribution of contribution of each constraint to the objective function is possible by calling GetHappyness().
    
    # Types description
    5 types of constraints are currently available.
    
    In the following, X will represent the allocation matrix which value is 1 iff a person i is allocated office j, 0 otherwise.
    In allocation models, this matrix usually have the constraint to have a single 1 per line and per column.
    
    R will reprensent the ressource properties matrix. R_jr = 1 iff the office j has the ressource r.
    This matrix usually represent the group in which a ressource belong. 
    In particular for the office allocation to worker, this corresponds to the repartition of offices among rooms.
    
    W will represent the matrix representing the weights attributed by users to a given property.
    
    P will represent the users distribution over users properties.
    For example, it represents P_is =1 iff a user i belongs to a service s.
    
    ## prBin 
    
    In the prBin constraint, the user attributes a weight (preference) on having a ressources with the labelled property.
    A given ressources can be given an amount of fulfillment of the property. 
    Negative weights can be attributed to a ressource.
    
    The prBin option is representative of the case in which an user desires to be on an office next to a window.
        
    The objective function of this option corresponds the sum of weights of users times the level of of fitting of the their office within the property.
    o = sum( Wt . X . R ) (Wt . X . R is a (nUser x 1) matrix)
    
    constraints can be added as follow : (Wt.X.R)_i <= (>=) valBoud iff (Wt.X.R)_i!=0.
    The condition (Wt.X.R)_i!=0 means that this matrix element must be null whatever the value of X.
        
    The level of agreement between user preferences and final allocation (called happyness) is given by :
        h = Wt . X . R
    
    ## prCat : the user give a preference on one of serveral possible outcome of a categorical property.
        
        - prBinTag : the user has a preference of having a property within the structure to which its ressources belong.
        
        - ppCat
    """
    def __init__(self, typ, label
                 , maxWeights = False
                 , bound=0
                 , valBound=0
                 , roomTag=[]
                 , multi=False
                 , removeSelf = False
    ):
        self.acceptedTypes = ['ppBin', 'ppCat', 'prBin', 'prCat', 'prBinCat']
        if typ not in self.acceptedTypes : raise RuntimeError('Constraint : Wrong type for constraint. Accepted types : ' + ' '.join(self.acceptedTypes))
        self.__type = typ

        if label == '' : raise RuntimeError('Constraint : Empty label')
        self.label = label
        self.maxWeights=maxWeights
        self.bound = bound
        self.valBound = valBound
        self.roomTag = roomTag
        if not self.roomTag and self.__type in [ 'ppBin', 'ppCat' ] : raise RuntimeError( 'Constraint : Need roomTag for option ', self.label )
        self.wish = np.array([])
        self.dispo = np.array([])
        self.K = 2
        self.prodVars = None
        self.binVars = None
        self.multi = multi
        self.posDispo=None
        
        self.inLabel = 'in' + self.label[0].upper() + self.label[1:]
        self.weightLabel = 'weight' + self.label[0].upper() + self.label[1:]
        self.removeSelf = removeSelf

    def GetType(self) : return self.__type

    def GetObjVal(self) :
        """
        Return the value of the objective function.
        The content of wish, dispo and prodVars is described in the corresponding DefineConstraint.
        0 is returned if the constraint does not contribute to the objective function.
        - pp : returns the sum of all prodVars, which values are the products of weights whith their corresponding dispo.
        - prBinCat : returns the sum of of the matrix product of wish and dispo.
        In case of multi option, the weight corresponding to the self liking is removed.
        - pr : return the sum of all wish time dispo

        """
        if not self.maxWeights : return 0
        elif 'pp' in self.__type : return pulp.lpSum(self.prodVars )
        elif  self.__type == 'prBinCat' : return  np.dot(self.wish.T, self.dispo ).sum() - ( self.wish if self.removeSelf else 0 )
        elif 'pr' in self.__type : return np.multiply(self.wish, self.dispo).sum()
        else : return 0

    #==========
    def GetColumnsOption(self, data) :
        """
        Return the suffix of the columns which are to be used in multi mode.
        """
        indices = [ int(x.replace(self.label, '')) for x in data.columns if self.label in x and x.replace(self.label, '')!='' ]
        return indices


    # =============================================================================
    # 
    # DEFINECONSTRAINT
    # 
    # =============================================================================
    def DefineConstraint(self, placement, officeData, persoData ) :
        """
        Call the Define constraint correspoinding to the constraint type.
        DefineConstraint fills self.wish and self.dispo according to the requirement of the constraint type
        """
        if 'pp' in self.__type  : self.DefinePPConstraint( placement, officeData, persoData )
        elif self.__type == 'prBin' : self.DefinePRBinConstraint( placement, officeData, persoData )
        elif self.__type == 'prCat' : self.DefinePRCatConstraint( placement, officeData, persoData )
        elif self.__type == 'prBinCat' : self.DefinePRBinCatConstraint( placement, officeData, persoData )
        else : raise RuntimeError( 'DefineConstraint : Unknown type for Constraint : ', self.__type )

    def DefinePPConstraint(self, placement, officeData, persoData ) :
        """
        Calls dedicated functions to fill wish and dispo.
        Compute the K as a value impossible to reach for any sum of weights.
        Create the continuous pulp variables representing the product of wish and dispo.
        Create the pulp binary variables representing wether a dispo value is null or not.

        The persons which interact must belong to the same entity tagged by self.roomTag.
        
        """
        if 'Cat' in self.__type : self.DefinePPCatConstraint( placement, officeData, persoData )
        else : self.DefinePPBinConstraint( placement, officeData, persoData )
        s = self.wish.shape

        self.prodVars = pulp.LpVariable.matrix( self.label+'Max', (np.arange(s[0]), np.arange(s[1])), cat='Continuous' )
        self.binVars = pulp.LpVariable.matrix( self.label+'Bin', (np.arange(s[0]), np.arange(s[1])), cat='Binary' )
        labels = [self.weightLabel] if not self.multi else [ self.weightLabel+str(x) for x in self.GetColumnsOption(persoData)]
        self.K = max(np.fabs(persoData[labels].values).sum(),self.K)
        self.posDispo = pulp.LpVariable.matrix( self.label+'PosDispo', (np.arange(s[0]), np.arange(s[1])), cat='Continuous' )
        return self


    #==========
    def DefinePPBinConstraint( self, placement, officeData, persoData ) :
        """
        Fills the wish and dispo for ppBin constraint.
        
        wish is a ( 1, rooms ) matrix containing the combined weights regarding the property from people from the entity.
        
        dispo is a ( 1, rooms ) matrix containing the number of users with the property in the given entity.
        
        """

        officeFilter = pd.pivot_table(officeData.loc[:,self.roomTag], columns=self.roomTag, index=officeData.index, aggfunc=len).fillna(0)
        officeFilter = np.dot(placement, officeFilter)
    
        self.wish = np.array([np.dot( persoData[self.weightLabel].values.T, officeFilter )])

        self.dispo = persoData[self.inLabel]
        self.dispo = np.array([np.dot( self.dispo.T, officeFilter )])


    #==========
    def DefinePPCatConstraint(self, placement, officeData, persoData ) :
        """Fills the wish and dispo for pp constraints.
        
        The requested user categories may not match the available users categories. 
        Only the inner join of requested and available are considered.
        
        wish is a square matrix containing the summ of weights generated by the simultaneous presence of categories i and j in an entity.
        dispo is a square matrix (same dimension as wish), labelling the simultaneous presence of of categories i and j in the same entity.
        
        
        """
        if self.__type != 'ppCat' : raise RuntimeError('Constraint::DefinePPCatConstraint : not using proper type.')
        suffix = self.GetColumnsOption(persoData) if self.multi else ['']
        usedOptions = [ self.label + str(x) for x in suffix]
        commonLabels = sorted(list(set(persoData.loc[:,usedOptions].values.ravel()).intersection(persoData[self.inLabel].values)))

        #self property of each person
        persoInOption = pd.pivot_table( persoData.loc[:, [self.inLabel]], columns=[self.inLabel], index=persoData.index, aggfunc=len).fillna(0)
        persoInOption = persoInOption[commonLabels].values

        #weight for preferences of each person
        persoFilter = pd.DataFrame()
        for x in suffix :
            table = pd.pivot_table( persoData.loc[:,[self.weightLabel+ str(x),self.label + str(x)]], values=self.weightLabel+ str(x), columns=[self.label + str(x)], index=persoData.index, aggfunc='sum').fillna(0)
            persoFilter = persoFilter.add(table, fill_value=0)

        persoFilter = persoFilter.loc[:,commonLabels]

        officeFilter = pd.pivot_table(officeData.loc[:, self.roomTag], columns=self.roomTag, index=officeData.index, aggfunc=len).fillna(0)
        officeFilter = np.dot( placement, officeFilter.values )
        self.dispo = np.dot( persoInOption.T, officeFilter)
        self.wish = np.dot( persoFilter.values.T, officeFilter)

    #==========
    def DefinePRBinCatConstraint(self, placement, officeData, persoData ) :
        """
        Fills the wish and dispo for prBinCat constraints
        
        wish ( user x 1) contains the weight a user i attributes to having a ressource with the labelled property.
        
        dispo ( user x cat) contains the category which have been attributed to a user i.
        """
        self.wish = persoData.loc[:, self.label].values
        officeFilter = pd.pivot_table(officeData.loc[:,[self.label]], columns=self.label, index=officeData.index, aggfunc=len).fillna(0)

        self.dispo = np.dot( placement, officeFilter )
        
    #==========
    def DefinePRBinConstraint( self, placement, officeData, persoData ) :
        """
        Fills the wish and dispo for prBin.
        
        wish ( user x 1 ) contains the weight a user i attributes to having a ressource with the labelled property
        
        dispo ( user x 1 ) contains wether the user has been attributed the property
        """
        self.wish = persoData.loc[:, self.label].values
        self.dispo = np.dot(placement, officeData.loc[:, self.label])

    #==========
    def DefinePRCatConstraint( self, placement, officeData, persoData ) :
        """
        Fills the wish and dispo for prcat
        
        wish ( user x cat ) contains the weights a user attributes to any category
        
        dispo ( user x cat ) contains the weights from each category which have been attributed to each user.
        """
        suffix = self.GetColumnsOption(persoData) if self.multi else ['']

        #returns which property has the office
        officeFilter = pd.pivot_table(officeData.loc[:,[self.label]], columns=self.label, index=officeData.index, aggfunc=len).fillna(0)

        #return the weight that a person attribute to being allocaetd category i
       # persoFilter = pd.pivot_table(persoData, values=self.weightLabel, columns=self.label, index=persoData.index, aggfunc='sum').fillna(0)
        persoFilter = pd.DataFrame()
        for x in suffix :
            table = pd.pivot_table(persoData, values=self.weightLabel+str(x), columns=self.label+str(x), index=persoData.index, aggfunc='sum').fillna(0)
            persoFilter = persoFilter.add(table, fill_value=0)

        commonLabels = list(set(persoFilter.columns).intersection(officeFilter.columns))
        officeFilter = officeFilter.loc[:,commonLabels].values
        self.wish = persoFilter.loc[:,commonLabels].values

        #return the properties which have been allocated to each perso
        self.dispo = np.dot( placement, officeFilter )


    # =============================================================================
    # 
    # SETCONSTRAINT
    # 
    # =============================================================================
    def SetConstraint(self, model) :
        """
        Distribute the SetConstraint call to the dedicated function for the Constraint type
        """
        if 'pp' in self.__type : self.SetPPConstraint( model )
        elif self.__type == 'prBin' and self.bound!=0 : self.SetPRBinConstraint( model )
        elif self.__type == 'prCat' and self.bound != 0 : self.SetPRCatConstraint(model)
        elif self.__type == 'prBinCat' and self.bound != 0 : self.SetPRBinCatConstraint(model)
        elif self.bound == 0 : return
        else : raise RuntimeError( 'SetConstraint : Unknown type for Constraint : ', self.__type )

    #==========
    def SetPRBinCatConstraint( self, model ) :
        """
        Set the prBinCat constraint into the input model
        """
        tot = np.dot( self.wish.T, self.dispo )
        for val in tot :
            if not val : continue
            if self.bound>0 : model += val <= self.valBound
            elif self.bound<0 : model += val >= self.valBound

    #==========
    def SetPRBinConstraint(self, model ) :
        """
        Set the prBin constraint into the input model
        """
        tot = np.multiply(self.wish, self.dispo)
        for val in tot :
            if not val : continue
            if self.bound>0 : model += val <= self.valBound
            elif self.bound<0 : model += val >= self.valBound

    #==========
    def SetPRCatConstraint(self, model ) :
        """
        Set the prBin constraint into the input model
        """
        tot = np.multiply(self.wish, self.dispo)
        for line in tot :
            for val in line :
                if not val : continue
                if self.bound>0 : model += val <= self.valBound
                elif self.bound<0 : model += val >= self.valBound
  
    #==========
    def SetPPConstraint( self, model ) :
        s = self.wish.shape
        for i in range(s[0]) :
            for j in range(s[1]) :
                model += self.prodVars[i][j] <= 2 * self.K + self.wish[i][j] - 2*self.K * self.binVars[i][j]
                model += self.binVars[i][j] <= self.dispo[i][j]
                model += self.binVars[i][j] >= self.dispo[i][j]/self.K
                model += self.prodVars[i][j] <= self.K * self.dispo[i][j]
                model += self.prodVars[i][j] >= - self.K * self.dispo[i][j]
                model += self.prodVars[i][j] >= -2 * self.K + self.wish[i][j] + 2*self.K * self.binVars[i][j]
#                model += self.posDispo[i][j] + self.dispo[i][j] / self.K <= 0
#                model += self.posDispo[i][j] - self.dispo[i][j] / self.K >= 0 
                if self.bound>0 : model += self.prodVars[i][j] <= ( self.valBound - self.K )*self.binVars[i][j] + self.K
                elif self.bound<0 :  model += self.prodVars[i][j] >= (self.valBound + self.K)*self.binVars[i][j] - self.K


    # =============================================================================
    # 
    # GETHAPPYNESS
    # 
    # =============================================================================
    def GetHappyness( self, placement, officeData, persoData ) :
        """
        Return the level of happyness of each user regarding the constraint
        """
        if not self.maxWeights  : return 0
        x = ArrayFromPulpMatrix2D( placement )
        if self.__type == 'prBinCat' : return self.GetPRBinCatHappyness( x, officeData, persoData )
        elif 'pr' in self.__type : return self.GetPRHappyness( x, officeData, persoData )
        elif 'pp' in self.__type : return self.GetPPHappyness( x, officeData, persoData)

    #==========
    def GetPRBinCatHappyness( self, placement, officeData, persoData ) :
        """
        Return the level of happyness of users for prBinCat constraints
        """

        self.DefinePRBinCatConstraint( placement, officeData, persoData )
        persoFilter = persoData.loc[:,self.label]
        officeFilter = pd.pivot_table(officeData.loc[:,self.roomTag], columns=self.roomTag, index=officeData.index, aggfunc=len).fillna(0)

        self.wish = persoFilter
        self.dispo = np.dot( placement, officeFilter ).sum(0)
        self.dispo = np.dot( officeFilter, self.dispo.T )
        self.dispo = np.dot( placement, self.dispo )
        return np.multiply( self.wish, self.dispo ) - ( self.wish if self.removeSelf else 0 )

    #==========
    def GetPRHappyness( self, placement, officeData, persoData ) :
        if self.__type == 'prBin' :
            self.DefinePRBinConstraint(placement, officeData, persoData )
            return np.multiply( self.wish, self.dispo )
        else :
            self.DefinePRCatConstraint(placement, officeData, persoData )
            return np.multiply( self.wish, self.dispo ).sum(1)

    def GetPPHappyness( self, placement, officeData, persoData ) :

        officeFilter = pd.pivot_table(officeData.loc[:,self.roomTag], columns=self.roomTag, index=officeData.index, aggfunc=len).fillna(0)
        persoFilter = None 
        
        if self.__type == 'ppBin' : 
            self.wish = persoData.loc[:, [self.weightLabel]]
            persoFilter =np.array([ persoData[self.inLabel]]).T
        else :
            suffix = self.GetColumnsOption(persoData) if self.multi else ['']
            usedOptions = [ self.label + str(x) for x in suffix] 
            commonLabels = sorted(list(set(persoData[usedOptions].values.ravel()).intersection(persoData[self.inLabel].values)))

            self.wish = pd.DataFrame()
            for x in suffix :
                table = pd.pivot_table( persoData.loc[:,[self.weightLabel+ str(x),self.label + str(x)]], values=self.weightLabel+ str(x), columns=[self.label + str(x)], index=persoData.index, aggfunc='sum').fillna(0)
                self.wish = self.wish.add(table, fill_value=0)

            self.wish = self.wish[commonLabels].values
            persoFilter = pd.pivot_table( persoData.loc[:, [self.inLabel]], columns=[self.inLabel], index=persoData.index, aggfunc=len).fillna(0)
            persoFilter = persoFilter.loc[:,commonLabels]

        self.dispo = np.dot(placement, officeFilter )
        self.dispo = np.dot(self.dispo, officeFilter.T)
        self.dispo = np.dot( self.dispo, placement.T)
        self.dispo = np.dot( self.dispo, persoFilter)

        return np.multiply(self.wish, self.dispo).sum(1)


#===========
def ArrayFromPulpMatrix2D( x ) :
    nRows = len(x)
    nCols = len(x[0])

    result = np.zeros((nRows, nCols))

    for i  in range(nRows):
        for j in range(nCols) :
            result[i][j] = pulp.value(x[i][j])

    return result
#==========
def GetRoomIndex( data) :
    """
    returns the set of columns names which allows identification of a room within the given dataset
    """
    labels = ['etage', 'roomID']
    columns = data.columns

    result =[ l for l in labels if l in columns ]
    if not result : raise RuntimeError( 'No tagging possible for a room' )
    return result

#==========
def FillRandomPerso( data, persoID, options = [] ) :
    """

    Change the data according to random options

    Arguments:
        data {DataFrame} -- Data to modify
        persoID {int} -- Index of the person to change properties

    """

    nFilled = np.random.randint(1, len(options))
    choices = np.random.choice(options, size=nFilled, replace=False )

    for i, v in enumerate(choices) :
        data.loc[persoID,v] = len(options)-i

        if 'weightPerso' in v :
            nPerso = int(v.replace('weightPerso', ''))
            nameCol = 'perso%d'%nPerso
            friendID = np.random.randint(0, len(data))
            data.loc[persoID, nameCol] = data.loc[friendID,'Nom']

        if v=='weightEtage' : data.loc[persoID,'etage'] = np.random.randint(1, 3)

#==========
def CreatePersoData( nPerso=10,
                     properties = {},
                     preferences = ['clim', 'mur', 'passage', 'sonnerie', 'wc', 'weightEtage', 'window']
                     ) :
    """"

    Generate a dataframe simulating a distribution of perso characteristics

    Keyword Arguments:
        nPerso {int} -- Number of characters to generate (default: {10})
    """

    dicoPersos = {'Nom' : ['Dummy%d'%(i) for i in range(nPerso)]}

    persoData = pd.DataFrame( dicoPersos)

    for prop, value in properties.items() : persoData[prop] = np.random.choice(value, size=nPerso)

    for col in preferences + ['etage']: persoData[col]=0

    persoCol = [ 'perso%i'%int(pref.replace('weightPerso', '')) for pref in preferences if 'weightPerso' in pref]
    for col in persoCol : persoData[col]=''

    #simulate the preferences
    for iPerso in range(nPerso) : FillRandomPerso( persoData, iPerso, preferences)

    return persoData

#==========
def CreateOfficeData( nOffice=12, properties = {} ) :
    officeData = pd.DataFrame( { 'officeID': np.arange(0, nOffice)})
    for prop, value in properties.items() : officeData[prop] = np.random.choice(value, size=nOffice )
    officeData = officeData.set_index('officeID')
    return officeData

#==========
def GetCountPerOfficeProp( placements, officeData, persoData, officeTags=['roomID'], persoTags=['inService'] ) :
    """
    Return the number of occurences of persons with a given property (personTag) in offices with another property (officeTag)
    """
    persoFilter = pd.pivot_table(persoData.loc[:,persoTags], columns=persoTags, index=persoData.index, aggfunc=len).fillna(0)
    persoFilter = persoFilter.values.T
    officeFilter = pd.pivot_table(officeData.loc[:,officeTags], columns=officeTags, index=officeData.index, aggfunc=len).fillna(0)
    return np.dot( np.dot(persoFilter, placements), officeFilter )


#==========
def DrawOutput( repartition, officeCoord ) :

    img = Image.open("OfficeID.png")
    draw = ImageDraw.Draw(img)
#    # font = ImageFont.truetype(<font-file>, <font-size>)
#    font = ImageFont.truetype("sans-serif.ttf", 16)
    font = ImageFont.load_default()
#    # draw.text((x, y),"Sample Text",(r,g,b))
    for row in repartition.itertuples() :
        x = officeCoord.loc[row.office,'xImage']
        y = officeCoord.loc[row.office,'yImage']
        draw.text((x, y), str(row.Nom),(0,0,0),font=font)
    img.save('sample-out.png')

#==========
def NormalizePersoPref( persoData, options ) :
    """
    Normalise the input preference weights by the sum of their absolute value
    """
    for opt in options :
        s = np.fabs(persoData.loc[:,opt]).sum(1)
        persoData.loc[:,opt] = (persoData.loc[:,opt].T / s).T

#==========
def MultiplyWithFilter( weights, filt ) :
    filt[filt!=0] = 1
    return np.multiply(weights, filt)


#==========
def TestInput( data, options ) :
    """
    Test wether a dataset has the columns correspondig to options.
    """
    columns = data.columns
    return all( x in columns for x in options)


#==========
def PrintOptimResults( placement
                      , persoData
                      , officeData
                      , diversityTag=[]
                      , constTag=[]
                      ) :
    resultFrame = pd.DataFrame({'ID': persoData.index, 'Nom':persoData['Nom']}).set_index('ID')
    resultFrame['office']=-1

    x=ArrayFromPulpMatrix2D(placement)
    for i, iPerso in enumerate(persoData.index) :
        for j, iRoom in enumerate(officeData.index) :
            if x[i][j] : resultFrame.loc[iPerso, 'office'] = iRoom

    print('diversityTag : ', diversityTag)
    if diversityTag :
        Delta = GetCountPerOfficeProp( x, officeData, persoData, persoTags=diversityTag)
        Delta = np.minimum(Delta, np.ones(Delta.shape))
        print('diversityTags : ' + ' '.join(diversityTag)+'\n', Delta)
        print('diversityObjective : ', Delta.sum())

    labels = []
    for c in constTag :
        resultFrame[c.label] =  c.GetHappyness( x, officeData, persoData )
        labels.append(c.label)

    resultFrame['totHappyness'] = resultFrame.loc[:,labels].sum(axis=1)

    print(resultFrame)
    print('Attributions Bureaux')
    for row in resultFrame.itertuples() :
        print( '%s is given office %i with happyness %2.2f' % (row.Nom,row.office, row.totHappyness))

    resultFrame.to_csv('demenagement.csv')
    resultFrame[['xImage', 'yImage']] = officeData[['xImage', 'yImage']]
    DrawOutput( resultFrame, officeData[['xImage', 'yImage']] )

#==========
def RoomOptimisation( officeData, persoData,
                     diversityTag=[],
                     constTag=[],
                     minimize=True,
                     printResults=False,
                     ) :

    officeOccupancy = pulp.LpVariable.matrix("officeOccupancy" ,(list(persoData.index.values), list(officeData.index.values)), cat='Binary')


    doDiversity = diversityTag
    delta=np.array([])
    if  doDiversity :
        roomTag=GetRoomIndex(officeData)
        # Delta counts the number of person from each inService within a room
        Delta = GetCountPerOfficeProp( officeOccupancy, officeData, persoData, persoTags=diversityTag)

        # delta_sr = 1 iif a person from inService s belongs to room r
        # This variable is used to represent the diversity of inServices. The objective function will contain a sum of delta across inServices and rooms.
        # This variable values will be fully constrained by Delta
        nService = len( persoData.groupby(diversityTag) )
        nRooms = len( officeData.groupby( roomTag ) )
        delta = pulp.LpVariable.matrix("delta" ,(np.arange(nService), np.arange(nRooms) ) ,cat='Binary')

    for c in constTag  : c.DefineConstraint( officeOccupancy, officeData, persoData )

    #--------------------------------------------
    #Define the optimisation model
    model = pulp.LpProblem("Office setting maximizing happyness", pulp.LpMaximize)


    #Objective function :
    model += (
            np.sum(delta)
            + pulp.lpSum( c.GetObjVal() for c in constTag )
            )


    #Each perso is set once
    for s  in np.sum(officeOccupancy, axis=0) : model += s <= 1

    #Each room is set at least once
    for s in np.sum(officeOccupancy, axis=1) : model += s == 1


    if doDiversity :
        #Constraint of delta
        for s in range( nService ) :
            for j in range( nRooms ) :
                model += delta[s][j] >= Delta[s][j]/len(persoData)
                model += delta[s][j] <= Delta[s][j]


    for c in constTag  : c.SetConstraint( model )

    # Solve the maximisation problem
    if minimize :
        model.solve()
        if pulp.LpStatus[model.status] !='Optimal' : raise RuntimeError('Infeasible Model')

    if printResults :
        PrintOptimResults( officeOccupancy
                      , persoData
                      , officeData
                      , diversityTag
                      , constTag
                      )


        print('\n==========\n')
        print('model status : ', pulp.LpStatus[model.status] )
        print('objective : ', pulp.value(model.objective) )


    return model, officeOccupancy

#==========
class TestMultiplyWithFilter( unittest.TestCase ) :
    def test_result(self):
        x = np.array( [1, 2, 3] )
        f = np.array( [1, 0, 4] )
        y = MultiplyWithFilter(x, f )
        self.assertTrue(np.allclose( [1,0,3], y, rtol=1e-05, atol=1e-08))

#==========
class TestGetCountPerOfficeProp(unittest.TestCase):

    def test_result(self ) :
        perso = pd.DataFrame({'inService':['SI', 'SI', 'RH'], 'isTall':[0,0,0]})
        office = pd.DataFrame({'roomID':[0,0,0], 'isLeft':[0,0,0]})

        counts = GetCountPerOfficeProp( np.diag([1,1, 1]), office, perso)
        self.assertTrue(np.allclose(counts, [[1.],[2.]], rtol=1e-05, atol=1e-08))


#==========
class TestNormalizePersoPref(unittest.TestCase):
    def test_result(self ) :
        perso = pd.DataFrame( {'Nom':['Dum0', 'Dum1', 'Dum2' ],
                                   'weightPerso1' : [10,3,1],
                                   'weightPerso2' : [0, 0, 6],
                                   'perso1' : ['Dum1', 'Dum2', 'Dum0'],
                                   'perso2': ['', '', 'Dum1']
                                   })
        
        NormalizePersoPref(perso, [['weightPerso1', 'weightPerso2']])

        perso2 = pd.DataFrame( {'Nom':['Dum0', 'Dum1', 'Dum2' ],
                                   'weightPerso1' : [1,1,1/7],
                                   'weightPerso2' : [0, 0, 6/7],
                                   'perso1' : ['Dum1', 'Dum2', 'Dum0'],
                                   'perso2': ['', '', 'Dum1']
                                   })
        assert_frame_equal( perso, perso2 )

#==========
class TestTestInput( unittest.TestCase):
    def test_result(self) :
        data = pd.DataFrame({'dum0': [1]})
        options = ['dum0', 'dum1']

        self.assertFalse( TestInput(data, options))

        data['dum1']=0
        self.assertTrue( TestInput(data, options))

#==========
class TestConstraint( unittest.TestCase ):
    def setUp(self) :
        self.persoData = pd.DataFrame({'table':[1, 1, 0],
                                       'etage':[1,0,2],
                                       'weightEtage':[1,1,2],
                                       'inService':['SI','RH', 'SI'],
                                       'weightService':[3,8,6],
                                       'service':['SI', '', 'RH'],
                                       'perso0':['Dum1', 'Dum2', 'Dum0'],
                                       'perso1':['Dum2', '', 'Dum1'],
                                       'inPerso':['Dum0', 'Dum1', 'Dum2'],
                                       'weightPerso0':[3, 6, 8],
                                       'weightPerso1':[1, 2, 5],
                                       'inPhone':[0, 1, 1],
                                       'weightPhone':[-2, 0, 1],
                                       'etage1':[2, 0, 1],
                                       'weightEtage1':[3,6,1]
                                      } )
        self.persoData['inPerso1']=self.persoData['inPerso']
        self.persoData['inPerso0']=self.persoData['inPerso']
        self.persoData['etage0'] = self.persoData['etage'] 
        self.persoData['weightEtage0'] = self.persoData['weightEtage'] 

        self.officeData = pd.DataFrame({'table':[0,0,1], 'etage':[1, 1, 2], 'roomID':[0,0,0] } )
        self.placement = np.diag([1, 1, 1])

        self.pulpVars = pulp.LpVariable.matrix("placement" ,(np.arange(3), np.arange(3)),cat='Binary')
        self.model = pulp.LpProblem("Office setting maximizing happyness", pulp.LpMaximize)

        #SetConstraints
        for s  in np.sum(self.pulpVars, axis=0) : self.model += s <= 1
        for s in np.sum(self.pulpVars, axis=1) : self.model += s == 1

    #==========
    def test_GetColumnsOption(self) :
        cons = Constraint( 'ppCat', 'perso', True, roomTag=['table'] )
        self.assertTrue(np.allclose([0, 1], cons.GetColumnsOption(self.persoData) , rtol=1e-05, atol=1e-08))
        
    # =============================================================================
    # PRBINCAT
    # =============================================================================

    def test_DefinePRBinCatConstraint_resultInput(self) :
        self.persoData['table'] = [2, 3, 0]
        cons = Constraint( 'prBinCat', 'table', True, roomTag=['table'] )
        cons.DefinePRBinCatConstraint( self.placement, self.officeData, self.persoData )

        self.assertTrue(np.allclose([2,3, 0], cons.wish , rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose([[1, 0], [1, 0], [0, 1]], cons.dispo , rtol=1e-05, atol=1e-08))

        self.assertAlmostEqual(5.0, cons.GetObjVal() )

        hap = cons.GetPRBinCatHappyness(np.diag([1, 1, 1]), self.officeData, self.persoData )
        self.assertTrue(np.allclose([4,6, 0], hap , rtol=1e-05, atol=1e-08))


    def test_DefinePRBinCatConstraint_resultConsUp(self) :
        cons = Constraint( 'prBinCat', 'table', False, roomTag=['table'], bound=1, valBound=1 )
        cons.DefinePRBinCatConstraint( self.pulpVars, self.officeData, self.persoData )
        self.model+=cons.GetObjVal()
        cons.SetConstraint(self.model)
        self.model.solve()
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )

        self.assertAlmostEqual(0, pulp.value(cons.GetObjVal()) )
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(x[2][2], 0 )

    def test_DefinePRBinCatConstraint_resultConsDownMax(self) :
        cons = Constraint( 'prBinCat', 'table', True, roomTag=['table'], bound=-1, valBound=1 )
        cons.DefinePRBinCatConstraint( self.pulpVars, self.officeData, self.persoData )
        self.model+=cons.GetObjVal()
        cons.SetConstraint(self.model)
        self.model.solve()

        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        self.assertAlmostEqual(2, pulp.value(cons.GetObjVal()) )
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(x[2][2], 0 )

    def test_DefinePRBinCatConstraint_resultInfeas(self) :
        cons = Constraint( 'prBinCat', 'table', True, roomTag=['table'], bound=-1, valBound=3 )
        cons.DefinePRBinCatConstraint( self.pulpVars, self.officeData, self.persoData )
        cons.SetConstraint(self.model)
        self.model.solve()

        self.assertEqual(pulp.LpStatus[self.model.status], 'Infeasible' )

    # =============================================================================
    # PRBIN
    # =============================================================================

    def test_DefinePRBinConstraint_resultInput(self) :
        self.officeData.loc[1,'table']=1
        cons = Constraint( 'prBin', 'table', True )
        cons.DefinePRBinConstraint( self.placement, self.officeData, self.persoData )

        self.assertTrue(np.allclose([1,1, 0], cons.wish , rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose([0, 1, 1], cons.dispo , rtol=1e-05, atol=1e-08))

        self.assertAlmostEqual(1, cons.GetObjVal() )

        hap = cons.GetPRHappyness(np.diag([1, 1, 1]), self.officeData, self.persoData )
        self.assertTrue(np.allclose([0,1, 0], hap , rtol=1e-05, atol=1e-08))


    def test_DefinePRBinConstraint_resultConsUp(self) :
        cons = Constraint( 'prBin', 'table', False, bound=1, valBound=0 )
        cons.DefinePRBinConstraint( self.pulpVars, self.officeData, self.persoData )
        cons.SetConstraint(self.model)
        self.model.solve()
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )

        self.assertAlmostEqual(0, pulp.value(cons.GetObjVal()) )
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(x[2][2], 1 )

    def test_DefinePRBinConstraint_resultConsDownMax(self) :
        self.officeData.loc[1,'table']=1
        cons = Constraint( 'prBin', 'table', True, bound=-1, valBound=1 )
        cons.DefinePRBinConstraint( self.pulpVars, self.officeData, self.persoData )
        self.model+=cons.GetObjVal()
        cons.SetConstraint(self.model)
        self.model.solve()

        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        self.assertAlmostEqual(2, pulp.value(cons.GetObjVal() ))
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(x[2][0], 1 )

    def test_DefinePRBinConstraint_resultInfeas(self) :

        cons = Constraint( 'prBin', 'table', True, bound=-1, valBound=1 )
        cons.DefinePRBinConstraint( self.pulpVars, self.officeData, self.persoData )
        cons.SetConstraint(self.model)
        self.model.solve()

        self.assertEqual(pulp.LpStatus[self.model.status], 'Infeasible' )

    def test_DefinePRBinConstraint_resultNegDispo(self) :
        self.officeData.loc[1,'table']=-2
        cons = Constraint( 'prBin', 'table', True  )
        cons.DefinePRBinConstraint( self.pulpVars, self.officeData, self.persoData )
        self.model+=cons.GetObjVal()
        cons.SetConstraint(self.model)
        self.model.solve()

        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        self.assertAlmostEqual(1, pulp.value(cons.GetObjVal() ))
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(x[2][1], 1 )

    def test_DefinePRBinConstraint_resultNegNeg(self) :
        self.officeData.loc[1,'table']=-2
        self.persoData.loc[2,'table']=-3
        cons = Constraint( 'prBin', 'table', True  )
        cons.DefinePRBinConstraint( self.pulpVars, self.officeData, self.persoData )
        self.model+=cons.GetObjVal()
        cons.SetConstraint(self.model)
        self.model.solve()

        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        self.assertAlmostEqual(7, pulp.value(cons.GetObjVal() ))
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(x[2][1], 1 )

    # =============================================================================
    # PRCAT
    # =============================================================================

    def test_DefinePRCatConstraint_resultInput(self) :
        cons = Constraint( 'prCat', 'etage', True )
        cons.DefinePRCatConstraint( self.placement, self.officeData, self.persoData )

        self.assertTrue(np.allclose([[1, 0], [0, 0], [0, 2]], cons.wish , rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose([[1, 0], [1, 0], [0, 1]], cons.dispo , rtol=1e-05, atol=1e-08))

        self.assertAlmostEqual(3, cons.GetObjVal() )

        hap = cons.GetPRHappyness(self.placement, self.officeData, self.persoData )
        self.assertTrue(np.allclose([1,0, 2], hap , rtol=1e-05, atol=1e-08))


    def test_DefinePRCatConstraint_resultConsUp(self) :
        cons = Constraint( 'prCat', 'etage', False, bound=1, valBound=0 )
        cons.DefinePRCatConstraint( self.pulpVars, self.officeData, self.persoData )
        cons.SetConstraint(self.model)
        self.model.solve()
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )

        self.assertAlmostEqual(0, pulp.value(cons.GetObjVal()) )
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(x[0][2], 1 )

    def test_DefinePRCatConstraint_resultConsDownMax(self) :
        cons = Constraint( 'prCat', 'etage', True, bound=-1, valBound=1 )
        cons.DefinePRCatConstraint( self.pulpVars, self.officeData, self.persoData )
        self.model+=cons.GetObjVal()
        cons.SetConstraint(self.model)
        self.model.solve()

        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        self.assertAlmostEqual(3, pulp.value(cons.GetObjVal()) )
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(x[2][2], 1 )

    def test_DefinePRCatConstraint_resultMulti(self) :
        cons = Constraint( 'prCat', 'etage', True, multi=True )
        cons.DefinePRCatConstraint( self.pulpVars, self.officeData, self.persoData )
        self.model+=cons.GetObjVal()
        cons.SetConstraint(self.model)
        self.model.solve()
        
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(4, pulp.value(cons.GetObjVal()) )
        
        self.assertAlmostEqual(x[0][2], 1 )


    def test_DefinePRCatConstraint_resultInfeas(self) :

        cons = Constraint( 'prCat', 'etage', True, bound=-1, valBound=3 )
        cons.DefinePRCatConstraint( self.pulpVars, self.officeData, self.persoData )
        cons.SetConstraint(self.model)
        self.model.solve()

        self.assertEqual(pulp.LpStatus[self.model.status], 'Infeasible' )

        # =============================================================================
        #         PPCAT
        #
        # =============================================================================

    def test_DefinePPCatConstraint_resultInput(self) :
        cons = Constraint( 'ppCat', 'service', True, roomTag=['etage'] )
        cons.DefinePPConstraint(self.placement, self.officeData, self.persoData )

        self.assertTrue(np.allclose( [[0, 6], [3, 0]], cons.wish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [[1, 0], [1, 1]], cons.dispo, rtol=1e-05, atol=1e-08))

        self.assertAlmostEqual(3, cons.GetObjVal() )
        hap = cons.GetPPHappyness(self.placement, self.officeData, self.persoData )
        self.assertTrue(np.allclose([3,0, 0], hap , rtol=1e-05, atol=1e-08))

    def test_DefinePPCatConstraint_resultSelfLiking(self ) :
        #Self liking is accepted
        self.persoData = pd.DataFrame({'inService':['SI','RH'], 'weightService':[3,8], 'service':['RH', 'RH'] })
        self.officeData = pd.DataFrame({'roomID':[0,1]})
        cons = Constraint('ppCat', 'service', roomTag=['roomID'])
        cons.DefinePPCatConstraint( np.diag([1,1]), self.officeData, self.persoData )

        self.assertTrue(np.allclose( [[3, 8]], cons.wish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [[0, 1]], cons.dispo, rtol=1e-05, atol=1e-08))

    def test_resultDiffNumberRoomPerso(self ) :
        cons = Constraint('ppCat', 'service', roomTag=['roomID'])
        self.officeData = pd.DataFrame({'roomID':[0,0,0,0]})
        cons.DefinePPCatConstraint( np.array([[1,0,0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]), self.officeData, self.persoData )
        self.assertTrue(np.allclose( [[6], [3]], cons.wish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [[1], [2]], cons.dispo, rtol=1e-05, atol=1e-08))

    def test_DefinePPCatConstraint_resultMulti(self) :
        cons = Constraint( 'ppCat', 'perso', True, roomTag=['etage'], multi=True )
        cons.DefinePPConstraint(self.placement, self.officeData, self.persoData )

        self.assertTrue(np.allclose( [[0, 8], [3, 5], [7, 0]], cons.wish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [[1, 0], [1, 0], [0, 1]], cons.dispo, rtol=1e-05, atol=1e-08))

        self.assertAlmostEqual(3, cons.GetObjVal() )
        hap = cons.GetPPHappyness(self.placement, self.officeData, self.persoData )
        self.assertTrue(np.allclose([3,0, 0], hap , rtol=1e-05, atol=1e-08))

    def test_DefinePPCatConstraint_resultMax(self) :
        cons = Constraint( 'ppCat', 'perso0', True, roomTag=['etage'] )
        cons.DefinePPConstraint( self.pulpVars, self.officeData, self.persoData )
        self.model+=cons.GetObjVal()
        cons.SetConstraint(self.model)
        self.model.solve()
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )

        self.assertAlmostEqual(8, pulp.value(cons.GetObjVal()) )
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(x[1][2], 1 )

    def test_DefinePPCatConstraint_resultInf(self) :
        cons = Constraint( 'ppCat', 'perso0', True, bound=1, valBound=0, roomTag=['etage'] )
        cons.DefinePPConstraint( self.pulpVars, self.officeData, self.persoData )
        self.model+=cons.GetObjVal()
        cons.SetConstraint(self.model)
        self.model.solve()
        self.assertEqual(pulp.LpStatus[self.model.status], 'Undefined' )

    def test_DefinePPCatConstraint_resultConsUpMax(self) :
        cons = Constraint( 'ppCat', 'perso1', True, bound=1, valBound=1,roomTag=['etage'] )
        cons.DefinePPConstraint( self.pulpVars, self.officeData, self.persoData )
        self.model+=cons.GetObjVal()
        cons.SetConstraint(self.model)
        self.model.solve()
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )

        self.assertAlmostEqual(1, pulp.value(cons.GetObjVal()) )
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(x[1][2], 1 )

    def test_DefinePPCatConstraint_resultConsUp(self) :
        cons = Constraint( 'ppCat', 'perso1', True, bound=1, valBound=0,roomTag=['etage'] )
        cons.DefinePPConstraint( self.pulpVars, self.officeData, self.persoData )
        self.model+=cons.GetObjVal()
        cons.SetConstraint(self.model)
        self.model.solve()
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )

        self.assertAlmostEqual(0, pulp.value(cons.GetObjVal()) )
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(x[2][2], 1 )

    # =============================================================================
    # 
    # PPBIN
    # 
    # =============================================================================
    def test_DefinePPBinConstraint_resultInput(self) :
        cons = Constraint( 'ppBin', 'phone', roomTag=['etage'], maxWeights=True)
        cons.DefinePPBinConstraint( self.placement, self.officeData, self.persoData )

        self.assertTrue(np.allclose( [-2, 1], cons.wish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [1, 1], cons.dispo, rtol=1e-05, atol=1e-08))

        self.assertAlmostEqual(-1, cons.GetObjVal() )
        hap = cons.GetPPHappyness(self.placement, self.officeData, self.persoData )
        self.assertTrue(np.allclose([-2, 0, 1], hap , rtol=1e-05, atol=1e-08))

    #==========
    def test_DefinePPBinConstraint_resultNegDispo(self) :
        self.persoData['inPhone']*=-1
        cons = Constraint( 'ppBin', 'phone', roomTag=['etage'], maxWeights=True)
        cons.DefinePPBinConstraint( self.placement, self.officeData, self.persoData )

        self.assertTrue(np.allclose( [-2, 1], cons.wish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [-1, -1], cons.dispo, rtol=1e-05, atol=1e-08))

        self.assertAlmostEqual(1, cons.GetObjVal() )
        hap = cons.GetPPHappyness(self.placement, self.officeData, self.persoData )
        self.assertTrue(np.allclose([2, 0, -1], hap , rtol=1e-05, atol=1e-08))

    #==========
    def test_DefinePPBinConstraint_resultNon1Dispo(self) :
        self.persoData.loc[2,'inPhone']=2
        cons = Constraint( 'ppBin', 'phone', roomTag=['etage'], maxWeights=True)
        cons.DefinePPBinConstraint( self.placement, self.officeData, self.persoData )

        self.assertTrue(np.allclose( [-2, 1], cons.wish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [1, 2], cons.dispo, rtol=1e-05, atol=1e-08))
        self.assertAlmostEqual(0, pulp.value(cons.GetObjVal()) )
        hap = cons.GetPPHappyness(self.placement, self.officeData, self.persoData )
        self.assertTrue(np.allclose([-2, 0, 2], hap , rtol=1e-05, atol=1e-08))

    #==========
    def test_DefinePPBinConstraint_resultConsUp(self) :
        cons = Constraint( 'ppBin', 'phone', False, bound=1, valBound=0, roomTag=['etage'] )
        cons.DefinePPConstraint( self.pulpVars, self.officeData, self.persoData )
        cons.SetConstraint(self.model)
        self.model.solve()
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )

        self.assertAlmostEqual(0, cons.GetObjVal() )
        
        self.assertAlmostEqual(x[1][2], 1 )
        
    #==========
    def test_DefinePPBinConstraint_resultConsDownMax(self) :
        self.persoData.loc[2:'weightPhone']=3
        cons = Constraint( 'ppBin', 'phone', True, bound=-1, valBound=1, roomTag=['etage'] )
        self.model+=cons.GetObjVal()
        cons.DefinePPConstraint( self.pulpVars, self.officeData, self.persoData )
        cons.SetConstraint(self.model)
        self.model.solve()

        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        self.assertAlmostEqual(3, pulp.value(cons.GetObjVal()) )
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(x[0][2], 1 )

    #==========
    def test_DefinePPBinConstraint_resultInfeas(self) :
        self.persoData.loc[0,'inPhone']=1
        cons = Constraint( 'ppBin', 'phone', True, bound=-1, valBound=1, roomTag=['etage'] )
        cons.DefineConstraint( self.pulpVars, self.officeData, self.persoData )

        cons.SetConstraint(self.model)
        self.model.solve()

        self.assertEqual(pulp.LpStatus[self.model.status], 'Undefined' )


#==========
class TestRoomOptimisation( unittest.TestCase ):
    def setUp(self) :
        self.persoData = pd.DataFrame({'inService':['SI','SI','RH'],
                                       'window':[1,0,0],
                                       'mur':[0, 0.5, 0],
                                       'etage':[1, 1, 1],
                                       'service':['RH', 'SI', ''],
                                       'weightService':[3,6,2],
                                       'inPhone':[0, 1, 1],
                                       'weightPhone':[-2, 0, 1],
                                       'weightEtage':[1, 0, 0.5],

                                       })
        self.officeData =pd.DataFrame({'roomID':[0,0,0],
                                       'window':[1, 0, 0],
                                       'mur':[0, 1, 0],
                                       'etage':[1, 2, 1],
                                       })

    def test_empty(self) :
        model, placement = RoomOptimisation( self.officeData, self.persoData)

        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), None )

        y = ArrayFromPulpMatrix2D( np.array(placement) )
        self.assertTrue(np.allclose(y.sum(1), [1,1,1], rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose(y.sum(0), [1,1,1], rtol=1e-05, atol=1e-08))


    def test_resultObjectiveDiversity(self) :
        self.officeData.drop('etage', inplace=True, axis=1)
        model, placement = RoomOptimisation( self.officeData, self.persoData, diversityTag=['inService'] )

        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 2 )

        y = ArrayFromPulpMatrix2D( np.array(placement) )
        self.assertTrue(np.allclose(y.sum(1), [1,1,1], rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose(y.sum(0), [1,1,1], rtol=1e-05, atol=1e-08))


    def test_resultDiversity(self) :
        self.officeData['roomID']=[0,1,1]
        self.officeData.drop('etage', inplace=True, axis=1)
        model, placement = RoomOptimisation( self.officeData, self.persoData, diversityTag=['inService'] )

        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 3 )
        y = ArrayFromPulpMatrix2D( np.array(placement) )

        room2 = y[:2,1:].sum(1)
        self.assertEqual(np.linalg.det(np.array([y[:2,0], room2])), 1 )

    def test_resultBinSpat(self) :
        constTag = [ Constraint( 'prBin', 'mur', True ), Constraint('prBin', 'window', True)]
        model, placement = RoomOptimisation( self.officeData, self.persoData, constTag=constTag )

        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 1.5 )
        y = ArrayFromPulpMatrix2D( np.array(placement) )

        self.assertTrue(np.allclose(y, np.diag([1,1,1]), rtol=1e-05, atol=1e-08))

    def test_resultCatSpat(self) :
        constTag = [Constraint('prCat', 'etage', True )]
        model, placement = RoomOptimisation( self.officeData, self.persoData, constTag=constTag )

        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 1.5 )
        self.assertEqual( placement[1][1].varValue, 1)

    def test_resultBinSpatNegWeight(self) :
        persoData = pd.DataFrame({'window':[1,0], 'mur':[0, -0.5]})
        officeData = pd.DataFrame({'window':[1, 0], 'mur':[0, 1]})
        constTag = [ Constraint( 'prBin', 'mur', True ), Constraint( 'prBin', 'window', True)]
        model, placement = RoomOptimisation( officeData, persoData, constTag=constTag )

        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 0.5 )
        y = ArrayFromPulpMatrix2D( np.array(placement) )

        self.assertTrue(np.allclose(y, np.diag([1,1]), rtol=1e-05, atol=1e-08))

    def test_resultPPCatMatching(self) :
        spatialTag = [Constraint('ppCat', 'service', maxWeights=True, roomTag=['roomID'] ) ]
        model, placement = RoomOptimisation( self.officeData, self.persoData, constTag=spatialTag )

        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 9 )

    def test_resultPPCatMatching0Weight(self) :
        self.persoData.loc[2,'weightService'] = 0
        constTag = [Constraint('ppCat', 'service', maxWeights=True , roomTag=['roomID'] ) ]
        model, placement = RoomOptimisation( self.officeData, self.persoData, constTag=constTag)

        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 9 )

    def test_resultPPCatMatchingNegWeight(self) :
        self.persoData['weightService']=[-3, 0, -6]
        self.persoData['service'] = ['SI', 'RH', 'SI']
        constTag = [Constraint('ppCat', 'service', maxWeights=True , roomTag=['roomID']) ]
        model, placement = RoomOptimisation( self.officeData, self.persoData, constTag=constTag )

        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), -9 )

    def test_resultPPCatMatchingAloneRoom(self) :
        #Check self liking
        persoData = pd.DataFrame({'inService':['RH','SI'], 'service':['SI','SI'], 'weightService':[3,6]})
        officeData = pd.DataFrame({'roomID':[0,1]})
        constTag = [Constraint('ppCat', 'service', maxWeights=True, roomTag=['roomID'] )]
        model, placement = RoomOptimisation( officeData, persoData, constTag=constTag )
        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 6 )

    def test_resultPPBinUpBound(self):
        self.officeData.loc[0,'roomID']=1
        self.persoData['weightPhone']=[1, 0, 0]
        constTag = [Constraint('ppBin', 'phone', maxWeights=False, bound=1, valBound=0, roomTag=['roomID'] )]
        model, placement = RoomOptimisation( self.officeData, self.persoData, constTag=constTag )

        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), None )

        y = ArrayFromPulpMatrix2D( np.array(placement) )
        self.assertEqual(y[0][0], 1)

    def test_resultPPBinMatching(self) :
        self.officeData['roomID'] = [1,0,0]
        constTag = [Constraint('ppBin', 'phone', maxWeights=True, roomTag=['roomID']  )]
        model, placement = RoomOptimisation( self.officeData, self.persoData, constTag=constTag)
        y = ArrayFromPulpMatrix2D( np.array(placement) )

        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 1 )
        self.assertEqual(y[0][0], 1.)

    def test_resultPRConstBin(self) :
        constTag = [Constraint('prBin', 'window', bound=1, valBound=0, roomTag=['roomID'])]
        model, placement = RoomOptimisation( self.officeData, self.persoData, constTag=constTag )
        y = ArrayFromPulpMatrix2D( np.array(placement) )

        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), None )
        self.assertEqual(y[0][0], 0)

    def test_resultPRConstCat(self) :
        self.persoData['etage']=[2, 1, 2]
        constTag = [ Constraint('prCat', 'etage', bound=1, valBound=0, roomTag=['roomID'])]
        model, placement = RoomOptimisation( self.officeData, self.persoData, constTag=constTag )
        y = ArrayFromPulpMatrix2D( np.array(placement) )

        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), None )
        self.assertEqual(y[2][1], 0)

#==========
if __name__ == '__main__':
    unittest.main()
