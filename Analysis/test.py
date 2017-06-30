import analysis

if __name__ == '__main__':
    A = analysis.Analysis2(shots=159243,time_windows=[300, 500])
    A.save("TEST_PICKLE.PICKLE")
    B = analysis.Analysis2.restore("TEST_PICKLE.PICKLE")
    print(A)
    print(B)
    
