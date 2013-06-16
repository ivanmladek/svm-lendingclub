
    #Optimal number of features : 4
    #Feature ranking:
    #1. feature 9 (0.233677) #number of inquires
    #2. feature 1 (0.193933) # FICO high
    #3. feature 10 (0.090055) #Revolving balance
    #4. feature 5 (0.080833) #DTI
    #5. feature 8 (0.076548) #Total_acc
    #6. feature 7 (0.065637) #Open account
    #7. feature 2 (0.063432) #ann_income
    #8. feature 4 (0.061500) #loan_amount
    #9. feature 0 (0.061454) #len(description)
    #10. feature 6 (0.049553) #purpose
    #11. feature 21 (0.017243) #Delinquences 2year
    #12  feature 15 (0.006136) #Bankruptcies
    #13. feature 12 (0.000000)
    #14. feature 11 (0.000000)
    #15. feature 20 (0.000000)
    #16. feature 13 (0.000000)
    #17. feature 14 (0.000000)
    #18. feature 16 (0.000000)
    #19. feature 17 (0.000000)
    #20. feature 3 (0.000000)
    #21. feature 18 (0.000000)
    #22. feature 19 (0.000000)


  training_data = np.array([[len(str(desc)), #0
                f, #1
                parse_finite(ann_inc), #2
                parse_finite(term), #3
                parse_finite(amount), #4
                parse_finite(dti), #5
                parse_finite(PURPOSE_DICT[purpose]),#6
                parse_finite(open_acc), #7
                parse_finite(total_acc), #8
                parse_finite(num_inq), #9
                parse_finite(revol_bal), #10
                parse_finite(parse_percent(revol_util)),#11
                parse_finite(parse_percent(apr)),#12
                parse_finite(total_balance), #13
                parse_finite(default120), #14
                parse_finite(bankruptcies), #15
                parse_finite(tot_coll_amnt), #16
                parse_finite(rev_gt0), #17
                parse_finite(rev_hilimit), #18
                parse_finite(oldest_rev), #19
                parse_finite(pub_rec), #20
                parse_finite(delinq_2) #21
                ]
                              for term, desc,purpose, f, ann_inc,amount,dti,
                              open_acc,total_acc, num_inq, revol_bal,revol_util, apr,
                              emp_length,
                              total_balance,default120,
                              bankruptcies,
                              tot_coll_amnt,
                              rev_gt0,
                              rev_hilimit,
                              oldest_rev,
                              pub_rec,
                              delinq_2,
                              list_d
                              in zip(finite.term,
                                     finite.desc,
                                     finite.purpose,
                                     finite.fico_range_high,
                                     finite.annual_inc,
                                     finite.loan_amnt,
                                     finite.dti,
                                     finite.open_acc,
                                     finite.total_acc,
                                     finite.inq_last_6mths,
                                     finite.revol_bal,
                                     finite.revol_util,
                                     finite.apr,
                                     finite.emp_length,
                                     finite.total_bal_ex_mort,
                                     finite.num_accts_ever_120_pd,
                                     finite.pub_rec_bankruptcies,
                                     finite.tot_coll_amt,
                                     finite.num_rev_tl_bal_gt_0,
                                     finite.total_rev_hi_lim,
                                     finite.mo_sin_old_rev_tl_op,
                                     finite.pub_rec_gt_100,
                                     finite.delinq_2yrs,
                                     finite.list_d,)
                              if ( parse_year(list_d) in target_y)])
