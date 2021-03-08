#!/usr/bin/env python
# coding: utf-8

# In[ ]:
try:
    import warnings
    warnings.filterwarnings('ignore')
    import pandas as pd
    import streamlit as st
    from PIL import Image
    import seaborn as sns
    import os

    #image file "ecommerce-new-banner.jpg"
    image = Image.open(os.path.join('ecommerce-new-banner.jpg'))
    st.image(image)

    url = "https://res.cloudinary.com/dqlh9q2iv/raw/upload/v1615221293/Transaction1_ogvyzq_1_safw4c.csv"
    df = pd.read_csv(url, sep=",")
    df.drop(["Unnamed: 0","isFlaggedFraud"], axis=1, inplace = True)

    st.title('E-Commerce Fraud Detection')
    '-------------------------------------------------------------'
    st.write("""
    ### This interface is based on the theme Automation. It traces the unauthorized deception in Ecommerce transactions on the basis of paysim1 data.###
    """)


    if st.button("Check parent dataset"):
        st.write("#### Source: https://www.kaggle.com/ntnu-testimon/paysim1 ####")
        st.write(df)
        st.write("### Parent set fraud data count ###")
        st.write(df['isFraud'].value_counts())
    else:
        pass




    '-------------------------------------------------------------'

    data = df.drop(["step","nameOrig","nameDest"], axis =1)
    data['type'] = data['type'].map({"PAYMENT":0,"CASH_OUT":1,"CASH_IN":2,"TRANSFER":3,"DEBIT":4})



    st.write("""
    ### Questionnaire ###
    """)


    image = Image.open(os.path.join('26906922.jpg'))
    st.sidebar.image(image)
    st.sidebar.write("Transaction Fraud occurs when a stolen payment card or data is used to generate an unauthorized transaction. The move to real-time transactions is causing significant security challenges for banks, merchants and issuers alike. Quicker transaction times increase the chances of fraudulent transactions going undetected.")
    st.sidebar.write("""
    ### Tips for Safe Online Transactions-
        1. Use advanced anti-malware program.
        2. Watch out for security vulnerabilities in your PC.
        3. Make sure you are using a secure connection.
        4. Deal with reputed websites only.
        5. Use credit cards for online shopping.
        6. Do not use public computers.
        7. Set a strong and complex password. ###
    """)


    model = "ML - Random Forest"
    A = 0.30
    B = 0



    q1 = st.text_input("Transaction type: ")
    q2 = float(st.text_input("Amount of payment: "))
    q3 = float(st.text_input("Sender pre-transaction Balance: "))
    q4 = float(st.text_input("Sender post-transaction Balance: "))
    q5 = float(st.text_input("Reciever pre-transaction Balance: "))
    q6 = float(st.text_input("Reciever post-transaction Balance: "))


    q1 = q1.lower()
    if q1 == "payment":
        q1 = 0
    elif q1 == "cash out":
        q1 = 1
    elif q1 == "cash in":
        q1 = 2
    elif q1 == "transfer":
        q1 = 3
    elif q1 == "debit":
        q1 = 4

    outp = [[q1,q2,q3,q4,q5,q6]]



    X = data.drop(["isFraud"],axis=1)
    y = data["isFraud"]

    from sklearn.model_selection import train_test_split
    X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = A ,random_state = B)
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    if model == "ML - Random Forest":

        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier(criterion="gini",n_estimators = 1000,random_state=B)
        rfc.fit(X_train, y_train)

        y_pred = rfc.predict(X_test)

        out = rfc.predict(outp)

        if st.button('Evaluate'):

            if out == [1]:
                st.write(out)
                st.error("Transaction Fraud Detected!")


            elif out == [0]:
                st.write(out)
                st.success("Transaction Safe!")
        else:
            pass


    if model == "ML - Decision Tree":

        from sklearn.tree import DecisionTreeClassifier
        dtc = DecisionTreeClassifier(criterion='gini')
        dtc.fit(X_train, y_train)

        y_pred = dtc.predict(X_test)

        out = dtc.predict(outp)

        if st.button('Evaluate'):

            if out == [1]:
                st.write(out)
                st.error("Transaction Fraud Detected!")


            elif out == [0]:
                st.write(out)
                st.success("Transaction Safe!")
        else:
            pass


    if model == "DL - SingleLayer Perceptron":

        from sklearn.linear_model import Perceptron
        clf = Perceptron(tol=1e-1, random_state=B,alpha=0.0025)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        out = clf.predict(outp)

        if st.button('Evaluate'):

            if out == [1]:
                st.write(out)
                st.error("Transaction Fraud Detected!")


            elif out == [0]:
                st.write(out)
                st.success("Transaction Safe!")
        else:
            pass


    if model == "DL - MultiLayer Perceptrons":

        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(hidden_layer_sizes=(128,128),max_iter=100, random_state=B)
        mlp.fit(X_train, y_train)

        y_pred = mlp.predict(X_test)

        out = mlp.predict(outp)

        if st.button('Evaluate'):

            if out == [1]:
                st.write(out)
                st.error("Transaction Fraud Detected!")


            elif out == [0]:
                st.write(out)
                st.success("Transaction Safe!")
        else:
            pass

    st.title("Reference")
    '------------------------------------------------------'
    st.write("""
    ## **Analysed and Designed by OpenSource Sages** """)
    '* Anurag Sen (@AnuragSen) [Admin]'
    '* Tanmay Padhi (@Wolverine)'
    '* Partha De (@PARTHA DE ⊿◤)'
    '* Kanishk Deoras(@kilodelta)'
    '* Arindam Datta (@HS108_PS5_arindam)'



except ValueError:
    pass
