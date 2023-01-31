* Solve the problem A (fulfill the utility_matrix)

    * our custom algorithm
      * Initialize the data (calculate and save the result_set data)
      ```
      python query\_rec.py init
      ```
      * Run the algorithm
      ```
      python query\_rec.py query_recommendation
      ```
    * als algorithm
      * Run the algorithm
      ```
      python als.py als
      ```

* Evaluation
  * Initialize the file for the evaluation (masked utility_matrix)
    * evaluate our custom algorithm
      ```
      python query\_rec.py query_recommendation_evaluate
      ```
    * evaluate als algorithm
      ```
      python als.py als_evaluate
      ```
    
