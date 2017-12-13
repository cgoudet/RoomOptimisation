import numpy as np
import pandas as pd
import pulp
import unittest
from pandas.util.testing import assert_frame_equal
from PIL import Image, ImageFont, ImageDraw


class Constraint() :
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
        
        self.inLabel = 'in' + self.label[0].upper() + self.label[1:]
        self.weightLabel = 'weight' + self.label[0].upper() + self.label[1:]
        self.removeSelf = removeSelf
        
    def GetType(self) : return self.__type
    
    def GetObjVal(self) : 
        if not self.maxWeights : return 0
        elif 'pp' in self.__type : return pulp.lpSum(self.prodVars )
        elif  self.__type == 'prBinCat' : return  np.dot(self.wish.T, self.dispo ).sum() - ( self.wish if self.removeSelf else 0 )
        elif 'pr' in self.__type : return np.multiply(self.wish, self.dispo).sum()
        else : return 0
    
    def DefineConstraint(self, placement, officeData, persoData ) :
        if 'pp' in self.__type  : self.DefinePPConstraint( placement, officeData, persoData )
        elif self.__type == 'prBin' : self.DefinePRBinConstraint( placement, officeData, persoData )
        elif self.__type == 'prCat' : self.DefinePRCatConstraint( placement, officeData, persoData )
        elif self.__type == 'prBinCat' : self.DefinePRBinCatConstraint( placement, officeData, persoData )
        else : raise RuntimeError( 'DefineConstraint : Unknown type for Constraint : ', self.__type )
        
    def DefinePPConstraint(self, placement, officeData, persoData ) : 
        if 'Cat' in self.__type : self.DefinePPCatConstraint( placement, officeData, persoData )
        else : (self.wish, self.dispo) = GetPPBinMatching( placement, officeData, persoData, [self.label], roomID=self.roomTag )
        s = self.wish.shape

        self.prodVars = pulp.LpVariable.matrix( self.label+'Max', (np.arange(s[0]), np.arange(s[1])), cat='Continuous' )
        self.binVars = pulp.LpVariable.matrix( self.label+'Bin', (np.arange(s[0]), np.arange(s[1])), cat='Binary' )
        labels = [self.weightLabel] if not self.multi else [ self.weightLabel+str(x) for x in self.GetColumnsOption(persoData)]
        self.K = max(np.fabs(persoData[labels].values).sum(),self.K)
        return self
    
    #==========
    def GetColumnsOption(self, data) :
        indices = [ int(x.replace(self.label, '')) for x in data.columns if self.label in x]
        return indices
        
        
    #==========
    def DefinePPCatConstraint(self, placement, officeData, persoData ) :

        suffix = self.GetColumnsOption(persoData) if self.multi else [''] 
        usedOptions = [ self.label + str(x) for x in suffix]
        commonLabels = sorted(list(set(persoData[usedOptions].values.ravel()).intersection(persoData[self.inLabel].values)))

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


    def DefinePRBinCatConstraint(self, placement, officeData, persoData ) :
        self.wish = persoData.loc[:, self.label].values
        officeFilter = pd.pivot_table(officeData.loc[:,[self.label]], columns=self.label, index=officeData.index, aggfunc=len).fillna(0)

        self.dispo = np.dot( placement, officeFilter )
        
    def DefinePRBinConstraint( self, placement, officeData, persoData ) :
        self.wish = persoData.loc[:, self.label].values
        self.dispo = np.dot(placement, officeData.loc[:, self.label])

        
    def DefinePRCatConstraint( self, placement, officeData, persoData ) :
        weightName = 'weight' + self.label[0].upper() + self.label[1:]
    
        #returns which property has the office
        officeFilter = pd.pivot_table(officeData.loc[:,[self.label]], columns=self.label, index=officeData.index, aggfunc=len).fillna(0)
    
        #return the weight that a person attribute to being allocaetd category i
        persoFilter = pd.pivot_table(persoData, values=weightName, columns=self.label, index=persoData.index, aggfunc='sum').fillna(0)
    
        commonLabels = list(set(persoFilter.columns).intersection(officeFilter.columns))
        officeFilter = officeFilter.loc[:,commonLabels].values
        self.wish = persoFilter.loc[:,commonLabels].values
       
        #return the properties which have been allocated to each perso
        self.dispo = np.dot( placement, officeFilter )
    


    def SetConstraint(self, model) :
        if 'pp' in self.__type : SetPPConstraint( model, self.wish, self.dispo, self.prodVars, self.binVars, self.K, self.bound, self.valBound )
        elif self.__type == 'prBin' and self.bound!=0 : self.SetPRBinConstraint( model )
        elif self.__type == 'prCat' and self.bound != 0 : self.SetPRCatConstraint(model)
        elif self.__type == 'prBinCat' and self.bound != 0 : self.SetPRBinCatConstraint(model)
        #else : raise RuntimeError( 'SetConstraint : Unknown type for Constraint : ', self.__type )
        
    def SetPRBinCatConstraint( self, model ) :
        tot = np.dot( self.wish.T, self.dispo )
        for val in tot :
            if not val : continue
            if self.bound>0 : model += val <= self.valBound
            elif self.bound<0 : model += val >= self.valBound

    def SetPRBinConstraint(self, model ) : 
        tot = np.multiply(self.wish, self.dispo)
        for val in tot :
            if not val : continue
            if self.label == 'mur' :
                print( 'mur constraint : ', val >= self.valBound )
            if self.bound>0 : model += val <= self.valBound
            elif self.bound<0 : model += val >= self.valBound
 
    def SetPRCatConstraint(self, model ) :
        tot = np.multiply(self.wish, self.dispo)
        for line in tot :
            for val in line :
                if not val : continue 
                if self.bound>0 : model += val <= self.valBound
                elif self.bound<0 : model += val >= self.valBound
                
                
    def GetHappyness( self, placement, officeData, persoData ) :
        if not self.maxWeights  : return 0
        x = ArrayFromPulpMatrix2D( placement )
        if self.__type == 'prBinCat' : return self.GetPRBinCatHappyness( x, officeData, persoData )
        elif 'pr' in self.__type : return self.GetPRHappyness( x, officeData, persoData )
        elif 'pp' in self.__type : return self.GetPPHappyness( x, officeData, persoData)
        
    def GetPRBinCatHappyness( self, placement, officeData, persoData ) : 

        self.DefinePRBinCatConstraint( placement, officeData, persoData )
        persoFilter = persoData.loc[:,self.label]
        officeFilter = pd.pivot_table(officeData.loc[:,self.roomTag], columns=self.roomTag, index=officeData.index, aggfunc=len).fillna(0) 
        
        self.wish = persoFilter
        self.dispo = np.dot( placement, officeFilter ).sum(0)
        self.dispo = np.dot( officeFilter, self.dispo.T )
        self.dispo = np.dot( placement, self.dispo )
        return np.multiply( self.wish, self.dispo ) - ( self.wish if self.removeSelf else 0 )
        
    def GetPRHappyness( self, placement, officeData, persoData ) : 
        if self.__type == 'prBin' : 
            self.DefinePRBinConstraint(placement, officeData, persoData )
            return np.multiply( self.wish, self.dispo )
        else : 
            self.DefinePRCatConstraint(placement, officeData, persoData )
            return np.multiply( self.wish, self.dispo ).sum(1)
    
    def GetPPHappyness( self, placement, officeData, persoData ) :
        
        officeFilter = pd.pivot_table(officeData.loc[:,self.roomTag], columns=self.roomTag, index=officeData.index, aggfunc=len).fillna(0)

        suffix = self.GetColumnsOption(persoData) if self.multi else [''] 
        usedOptions = [ self.label + str(x) for x in suffix]
        commonLabels = sorted(list(set(persoData[usedOptions].values.ravel()).intersection(persoData[self.inLabel].values)))


        #weight for preferences of each person
        persoFilter = pd.DataFrame()
        for x in suffix :
            table = pd.pivot_table( persoData.loc[:,[self.weightLabel+ str(x),self.label + str(x)]], values=self.weightLabel+ str(x), columns=[self.label + str(x)], index=persoData.index, aggfunc='sum').fillna(0)
            persoFilter = persoFilter.add(table, fill_value=0)
        
        if self.__type == 'ppBin' : self.wish = persoData.loc[:, [self.weightLabel]]
        else : 
            self.wish = pd.DataFrame()
            for x in suffix : 
                table = pd.pivot_table( persoData.loc[:,[self.weightLabel+ str(x),self.label + str(x)]], values=self.weightLabel+ str(x), columns=[self.label + str(x)], index=persoData.index, aggfunc='sum').fillna(0)
                self.wish = self.wish.add(table, fill_value=0)

            self.wish = self.wish[commonLabels].values
        
        self.dispo = np.dot(placement, officeFilter )
        self.dispo = np.dot(self.dispo, officeFilter.T)
        self.dispo = np.dot( self.dispo, placement.T)

        persoFilter = pd.pivot_table( persoData.loc[:, [self.inLabel]], columns=[self.inLabel], index=persoData.index, aggfunc=len).fillna(0)
        
        persoFilter = persoFilter.loc[:,commonLabels]
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
def GetPPBinSingleMatching( placement, officeData, persoData, option, roomID=[] ) :
    """
    Return the wishing and dispo arrays for a single matching between binary property
    """
    
    if not roomID : roomID = GetRoomIndex(officeData)

    weightName = 'weight' + option[0].upper() + option[1:]
    inName = weightName.replace('weight', 'in' )
    
    officeFilter = pd.pivot_table(officeData.loc[:,roomID], columns=roomID, index=officeData.index, aggfunc=len).fillna(0)
    officeFilter = np.dot(placement, officeFilter)
    
    weightPerRoom = np.dot( persoData[weightName].values.T, officeFilter )
    
    persoDispo = persoData[inName]
    persoDispo = np.dot( persoDispo.T, officeFilter )
    
    return (weightPerRoom, persoDispo)


        
#==========
def GetPPBinMatching( placement, officeData, persoData, properties, roomID=[] ) :
    totWish=[]
    totDispo=[]
    
    for opt in properties : 
        wish, dispo = GetPPBinSingleMatching(placement, officeData, persoData, opt, roomID)
        totWish.append(wish)
        totDispo.append(dispo)

    return (np.array(totWish), np.array(totDispo))


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
    for opt in options : 
        s = np.sqrt((persoData.loc[:,opt]**2).sum(1))
        persoData.loc[:,opt] = (persoData.loc[:,opt].T / s).T
        


        
#==========
def MultiplyWithFilter( weights, filt ) : 
    filt[filt!=0] = 1
    return np.multiply(weights, filt)


#==========
def GetPPCatMatching( placement, officeData, persoData, properties, roomID=[] ):
    result = [DefinePPCatConstraint( placement, officeData, persoData, opt, roomID ) for opt in properties ]
    return result   

#==========
def TestInput( data, options ) :
    """
    Test wether a dataset has the columns correspondig to options.
    """
    columns = data.columns
    return all( x in columns for x in options)

#==========
def SetPPConstraint( model, wish, dispo, pulpMaxVars, pulpBinVars, K, bound = 0, value=0) :
    s = wish.shape
    for i in range(s[0]) :
        for j in range(s[1]) : 
            model += pulpMaxVars[i][j] <= 2 * K + wish[i][j] - 2*K * pulpBinVars[i][j]
            model += pulpBinVars[i][j] <= dispo[i][j]
            model += pulpBinVars[i][j] >= dispo[i][j]/K
            model += pulpMaxVars[i][j] <= K * dispo[i][j]
            model += pulpMaxVars[i][j] >= - K * dispo[i][j]
            model += pulpMaxVars[i][j] >= -2 * K + wish[i][j] + 2*K * pulpBinVars[i][j]
            if bound>0 : model += pulpMaxVars[i][j] <= ( value - K )*pulpBinVars[i][j] + K
            elif bound<0 :  model += pulpMaxVars[i][j] >= (value + K)*pulpBinVars[i][j] - K

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
                                   'weightPerso1' : [1,1,1/np.sqrt(37)], 
                                   'weightPerso2' : [0, 0, 6/np.sqrt(37)],
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
class TestGetPPBinSingleMatching(unittest.TestCase):
    
    def test_resultNegWeight(self ) :
        perso = pd.DataFrame({'inPhone':[0, 1, 1], 'weightPhone':[-2, 0, 0] })
        office = pd.DataFrame({'roomID':[0,0,1]})     
        (wish, dispo) = GetPPBinSingleMatching( np.diag([1,1, 1]), office, perso, 'phone' )
        self.assertTrue(np.allclose( [-2, 0], wish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [1, 1], dispo, rtol=1e-05, atol=1e-08))
        
    def test_result(self ) :
        perso = pd.DataFrame({'inPhone':[0, 1, 1], 'weightPhone':[2, 0, 0] })
        office = pd.DataFrame({'roomID':[0,0,1]})     
        (wish, dispo) = GetPPBinSingleMatching( np.diag([1,1, 1]), office, perso, 'phone' )
        self.assertTrue(np.allclose( [2, 0], wish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [1, 1], dispo, rtol=1e-05, atol=1e-08))

#==========
class TestGetPPBinMatching(unittest.TestCase):
    
    def test_result(self ) :
        perso = pd.DataFrame({'inPhone':[0, 1, 1], 'weightPhone':[-2, 0, 0], 'inSmoke':[1, 0, 1], 'weightSmoke':[0, -5, 0] })
        office = pd.DataFrame({'roomID':[0,0,1]})     
        (wish, dispo) = GetPPBinMatching( np.diag([1,1, 1]), office, perso, ['phone', 'smoke' ])
        self.assertTrue(np.allclose( [[-2, 0],[-5, 0]], wish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [[1, 1],[1, 1]], dispo, rtol=1e-05, atol=1e-08))

    def test_resultSelfLiking(self ) :
        perso = pd.DataFrame({'inPhone':[0, 1, 1], 'weightPhone':[-2, 1, 0], 'inSmoke':[1, 0, 1], 'weightSmoke':[1, -5, 0] })
        office = pd.DataFrame({'roomID':[0,0,1]})     
        (wish, dispo) = GetPPBinMatching( np.diag([1,1, 1]), office, perso, ['phone', 'smoke' ])
        self.assertTrue(np.allclose( [[-1, 0],[-4, 0]], wish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [[1, 1],[1, 1]], dispo, rtol=1e-05, atol=1e-08))

#========== 
#==========
class TestSetConstraint(unittest.TestCase):
    def setUp( self ) : 
        self.persoData = pd.DataFrame({'inPhone':[0, 1, 1], 
                                       'weightPhone':[-2, 0, 1],
                                       'window' : [1, 0, 0],
                                       'clim' : [0, 0, 1],
                                       'etage' : [1, 0, 2],
                                       'weightEtage': [1, 1, 0]
                                        })
        self.officeData = pd.DataFrame({'roomID':[1,0,0],
                                        'window':[1, 0, 0],
                                        'clim':[0, 0, 1],
                                        'etage':[1, 0, 0],
                                        })    
        
        self.pulpVars = pulp.LpVariable.matrix("officeOccupancy" ,(np.arange(3), np.arange(3)),cat='Binary')
        self.model = pulp.LpProblem("Office setting maximizing happyness", pulp.LpMaximize)

        #SetConstraints
        for s  in np.sum(self.pulpVars, axis=0) : self.model += s <= 1
        for s in np.sum(self.pulpVars, axis=1) : self.model += s == 1    


        
        
    def test_PPConstraint(self) :
        tag='phone'
        
        (wish, dispo) = GetPPBinMatching(self.pulpVars, self.officeData, self.persoData, [tag] )
        s = wish.shape

        pulpMaxVars =  pulp.LpVariable.matrix( 'ppBin', (np.arange(s[0]), np.arange(s[1])), cat='Continuous' )
        K = np.fabs(self.persoData[['weight' + tag[0].upper() + tag[1:]]]).values.sum()
        pulpBinVars =  pulp.LpVariable.matrix( tag+'Bin', (np.arange(s[0]), np.arange(s[1])), cat='Binary' )
        
        self.model += pulp.lpSum(pulpMaxVars)
        
        SetPPConstraint( self.model, wish, dispo, pulpMaxVars, pulpBinVars, K ) 
        
        self.model.solve()        
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        self.assertEqual(pulp.value(self.model.objective), 1 )

    def test_PPConstraintUpBound(self) :
        tag='phone'
        self.persoData['weightPhone']=[1, 0, 0]

        (wish, dispo) = GetPPBinMatching(self.pulpVars, self.officeData, self.persoData, [tag], roomID=['roomID'] )
        s = wish.shape

        pulpMaxVars =  pulp.LpVariable.matrix( 'ppBin', (np.arange(s[0]), np.arange(s[1])), cat='Continuous' )
        K = max(np.fabs(self.persoData[['weight' + tag[0].upper() + tag[1:]]]).values.sum(), 2)
        pulpBinVars =  pulp.LpVariable.matrix( tag+'Bin', (np.arange(s[0]), np.arange(s[1])), cat='Binary' )
        
        self.model += pulp.lpSum(pulpMaxVars)
        
        SetPPConstraint( self.model, wish, dispo, pulpMaxVars, pulpBinVars, K , bound=1, value=0) 
        self.model.solve()  
        x = ArrayFromPulpMatrix2D(self.pulpVars)
        
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        self.assertEqual(pulp.value(self.model.objective), 0 )
        self.assertEqual(x[0][0], 1)

    def test_PPConstraintLowBound(self) :
        tag='phone'
        self.persoData['weightPhone']=[2,0,-1]
        self.persoData['inPhone']=[0, 1, 0]
        
        (wish, dispo) = GetPPBinMatching(self.pulpVars, self.officeData, self.persoData, [tag] )
        s = wish.shape

        pulpMaxVars =  pulp.LpVariable.matrix( 'ppBin', (np.arange(s[0]), np.arange(s[1])), cat='Continuous' )
        K = np.fabs(self.persoData[['weight' + tag[0].upper() + tag[1:]]]).values.sum()
        pulpBinVars =  pulp.LpVariable.matrix( tag+'Bin', (np.arange(s[0]), np.arange(s[1])), cat='Binary' )
                
        SetPPConstraint( self.model, wish, dispo, pulpMaxVars, pulpBinVars, K, bound=-1, value = 1 ) 
        
        self.model += pulp.lpSum(pulpMaxVars)

        self.model.solve()      
        x = ArrayFromPulpMatrix2D(self.pulpVars)

        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        self.assertEqual(pulp.value(self.model.objective), 2 )
        
        self.assertEqual(x[2][0], 1)
        
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
                                      } )
        
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
        cons.SetConstraint(self.model)
        self.model.solve()
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        
        self.assertAlmostEqual(0, cons.GetObjVal() )
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(x[2][2], 0 )

    def test_DefinePRBinCatConstraint_resultConsDownMax(self) :
        cons = Constraint( 'prBinCat', 'table', True, roomTag=['table'], bound=-1, valBound=1 )
        cons.DefinePRBinCatConstraint( self.pulpVars, self.officeData, self.persoData )
        cons.SetConstraint(self.model)
        self.model.solve()
        
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        self.assertAlmostEqual(2, cons.GetObjVal() )
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
        
        self.assertAlmostEqual(0, cons.GetObjVal() )
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(x[2][2], 1 )
        
    def test_DefinePRBinConstraint_resultConsDownMax(self) :
        self.officeData.loc[1,'table']=1
        cons = Constraint( 'prBin', 'table', True, bound=-1, valBound=1 )
        cons.DefinePRBinConstraint( self.pulpVars, self.officeData, self.persoData )
        cons.SetConstraint(self.model)
        self.model.solve()
        
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        self.assertAlmostEqual(2, cons.GetObjVal() )
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(x[2][0], 1 )
 
    def test_DefinePRBinConstraint_resultInfeas(self) :

        cons = Constraint( 'prBin', 'table', True, bound=-1, valBound=1 )
        cons.DefinePRBinConstraint( self.pulpVars, self.officeData, self.persoData )
        cons.SetConstraint(self.model)
        self.model.solve()
        
        self.assertEqual(pulp.LpStatus[self.model.status], 'Infeasible' )
        
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
        
        self.assertAlmostEqual(0, cons.GetObjVal() )
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(x[0][2], 1 )
        
    def test_DefinePRCatConstraint_resultConsDownMax(self) :
        cons = Constraint( 'prCat', 'etage', True, bound=-1, valBound=1 )
        cons.DefinePRCatConstraint( self.pulpVars, self.officeData, self.persoData )
        cons.SetConstraint(self.model)
        self.model.solve()
        
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        self.assertAlmostEqual(3, cons.GetObjVal() )
        x = ArrayFromPulpMatrix2D( self.pulpVars )
        self.assertAlmostEqual(x[2][2], 1 )
 
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
    
    def test_DefinePPCatConstraint_result(self ) :
        cons = Constraint('ppCat', 'service', roomTag=['roomID'])  
        cons.DefinePPCatConstraint( np.diag([1,1, 1]), self.officeData, self.persoData)
        self.assertTrue(np.allclose( [[6], [3]], cons.wish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [[1], [2]], cons.dispo, rtol=1e-05, atol=1e-08))
    
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
