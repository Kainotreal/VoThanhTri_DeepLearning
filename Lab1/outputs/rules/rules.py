def findDecision(obj): #obj[0]: x1, obj[1]: x2, obj[2]: x3, obj[3]: x4
   # {"feature": "x3", "instances": 100, "metric_value": 0.3397, "depth": 1}
   if obj[2]>2.0744597382947285:
      # {"feature": "x4", "instances": 68, "metric_value": 0.084, "depth": 2}
      if obj[3]<=1.6:
         # {"feature": "x1", "instances": 36, "metric_value": 0.054, "depth": 3}
         if obj[0]<=7.0:
            # {"feature": "x2", "instances": 35, "metric_value": 0.0538, "depth": 4}
            if obj[1]<=2.7:
               return 'Iris-versicolor'
            elif obj[1]>2.7:
               return 'Iris-versicolor'
            else:
               return 'Iris-versicolor'
         elif obj[0]>7.0:
            return 'Iris-virginica'
         else:
            return 'Iris-versicolor'
      elif obj[3]>1.6:
         # {"feature": "x1", "instances": 32, "metric_value": 0.0469, "depth": 3}
         if obj[0]>5.9:
            return 'Iris-virginica'
         elif obj[0]<=5.9:
            # {"feature": "x2", "instances": 4, "metric_value": 0.0, "depth": 4}
            if obj[1]<=3.0:
               return 'Iris-virginica'
            elif obj[1]>3.0:
               return 'Iris-versicolor'
            else:
               return 'Iris-virginica'
         else:
            return 'Iris-virginica'
      else:
         return 'Iris-versicolor'
   elif obj[2]<=2.0744597382947285:
      return 'Iris-setosa'
   else:
      return 'Iris-versicolor'
