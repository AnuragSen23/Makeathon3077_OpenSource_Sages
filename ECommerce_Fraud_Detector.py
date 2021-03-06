#!/usr/bin/env python
# coding: utf-8

# In[ ]:
try:
    import pandas as pd
    import streamlit as st
    import seaborn as sns
    import matplotlib.pyplot as plt
    from PIL import Image
    import os

    #image file "ecommerce-new-banner.jpg"
    image = Image.open(os.path.join('ecommerce-new-banner.jpg'))
    st.image(image)


    sid = 'https://res.cloudinary.com/dqlh9q2iv/raw/upload/v1615056529/ECommerce_new_bfhsrm.csv'
    df = pd.read_csv(sid, sep=",")
    df.drop(['Unnamed: 0'],axis=1,inplace=True)



    url = 'https://res.cloudinary.com/dqlh9q2iv/raw/upload/v1615033597/samples/Ecommerce_Fraud_dataset_ryx7ka.csv'
    data = pd.read_csv(url, sep=",")
    data.drop(['Unnamed: 0'],axis=1,inplace=True)

    st.title('E-Commerce Fraud Detection')
    '⊿◤-------------------------------------------------------------⊿◤'
    st.write("""
    ### This interface is based on the theme Automation. It traces the unauthorized deception in Ecommerce transactions on the basis of the data provided. It works on ML - Random Forest classifier framework. ###
    """)

    st.write(df)
    '⊿◤-------------------------------------------------------------⊿◤'
    st.write("""
    ### Questionnaire ###
    """)

    q1 = float(st.text_input("Enter the No of Transactions: "))
    q2 = float(st.text_input("Enter the No of Orders: "))
    q3 = float(st.text_input("Enter the No of Payments: "))
    q4 = float(st.text_input("Enter the Total transaction amount: "))
    q5 = float(st.text_input("Enter the No of transactions Failures: "))
    q6 = float(st.text_input("Enter the No of CardPayments: "))
    q7 = float(st.text_input("Enter the No of BitcoinPayments: "))
    q8 = float(st.text_input("Enter the No of OrdersFulfilled: "))
    q9 = float(st.text_input("Enter the  American Express CardPayments:"))
    q10 = float(st.text_input("Enter the VISA16 Payments: "))

    outp = [[q1,q2,q3,q4,q5,q6,q7,q8,q9,q10]]


    X = data.drop(["Fraud","Mastercard","JCB_16","JCB_15","DC_CB","ApplePayments","PaymentRegFail","Discover","Duplicate_IP","VISA_13","OrdersFailed","OrdersPending","PaypalPayments","Duplicate_Address","Trns_fail_order_fulfilled","Voyager","Maestro","Fraud_Decoded","index","customerEmail","customerPhone","customerDevice","customerIPAddress","customerBillingAddress"],axis=1)
    y = data['Fraud_Decoded']

    from sklearn.model_selection import train_test_split
    X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.30 ,random_state = 42)
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(criterion="gini",n_estimators = 1000,random_state=0)
    rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test)

    out = rfc.predict(outp)

    if st.button('Evaluate'):
        if out == [1]:
            '⊿◤-------------------------------------------------------------⊿◤'
            st.write(out)
            st.error("Transaction Fraud Detected!")


        elif out == [0]:
            '⊿◤-------------------------------------------------------------⊿◤'
            st.write(out)
            st.success("Transaction Safe!")
    else:
        pass


    st.title("Reference")
    '⊿◤------------------------------------------------------⊿◤'
    st.write("""
    ## **Analysed and Designed by OpenSource Sages** """)
    '* Anurag Sen (@AnuragSen) [Admin]'
    '* Tanmay Padhi (@Wolverine)'
    '* Partha De (@PARTHA DE ⊿◤)'
    '* Kanishk Deoras(@kilodelta)'
    '* Arindam Datta (@HS108_PS5_arindam)'



except ValueError:
    pass
