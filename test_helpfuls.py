import numpy as np 
import pytest
import helpful_functions 

hpf = helpful_functions.HelpfulFunctions()

def test_cart_sph():
    test1_point = [1,0,0]
    test2_point = [0,1,0]
    test3_point = [0,0,1]
    test4_point = [0,1,1]
    test5_point = [0,0,-1]
    test6_point = [0,-1,-1]
    test7_point = [-1,0, -1]
    test8_point = [12,0,-4321]

    test1 = hpf.cart_to_sph(test1_point[0], test1_point[1], test1_point[2])
    test2 = hpf.cart_to_sph(test2_point[0], test2_point[1], test2_point[2])
    test3 = hpf.cart_to_sph(test3_point[0], test3_point[1], test3_point[2])
    test4 = hpf.cart_to_sph(test4_point[0], test4_point[1], test4_point[2])
    test5 = hpf.cart_to_sph(test5_point[0], test5_point[1], test5_point[2])
    test6 = hpf.cart_to_sph(test6_point[0], test6_point[1], test6_point[2])
    test7 = hpf.cart_to_sph(test7_point[0], test7_point[1], test7_point[2])
    test8 = hpf.cart_to_sph(test8_point[0], test8_point[1], test8_point[2])


    assert test1 == (1,np.pi/2,0)
    assert test2 == (1, np.pi/2, np.pi/2)
    assert test3 == (1,0,0)
    assert test4 == (np.sqrt(2), np.pi/4, np.pi/2)
    assert test5 == (1, np.pi, 0)
    assert test6 == (np.sqrt(2), 3*np.pi/4, -np.pi/2)
    assert test7 == (np.sqrt(2), 3*np.pi/4, np.pi)
    assert test8 == (4321.016662777407, 3.1388155258068196, 0)


def test_sph_cart():
    test1_point = (1,np.pi/2,0)
    test2_point =  (1, np.pi/2, np.pi/2)
    test3_point = (1,0,0)
    test4_point = (np.sqrt(2), np.pi/4, np.pi/2)
    test5_point = (1, np.pi, 0)
    test6_point= (np.sqrt(2), 3*np.pi/4, -np.pi/2)
    test7_point = (np.sqrt(2), 3*np.pi/4, np.pi)
    test8_point = (4321.016662777407, 3.1388155258068196, 0)

    test1 = hpf.sph_to_cart(test1_point[0], test1_point[1], test1_point[2])
    test2 = hpf.sph_to_cart(test2_point[0], test2_point[1], test2_point[2])
    test3 = hpf.sph_to_cart(test3_point[0], test3_point[1], test3_point[2])
    test4 = hpf.sph_to_cart(test4_point[0], test4_point[1], test4_point[2])
    test5 = hpf.sph_to_cart(test5_point[0], test5_point[1], test5_point[2])
    test6 = hpf.sph_to_cart(test6_point[0], test6_point[1], test6_point[2])
    test7 = hpf.sph_to_cart(test7_point[0], test7_point[1], test7_point[2])
    test8 = hpf.sph_to_cart(test8_point[0], test8_point[1], test8_point[2])

    assert np.isclose(test1, [1,0,0]).all()
    assert np.isclose(test2, [0,1,0]).all()
    assert np.isclose(test3, [0,0,1]).all()
    assert np.isclose(test4,[0,1,1]).all()
    assert np.isclose(test5, [0,0,-1]).all()
    assert np.isclose(test6,[0,-1,-1]).all()
    assert np.isclose(test7, [-1,0, -1]).all()
    assert np.isclose(test8, [12,0,-4321]).all()


def test_unit_vector():
    test1_vec = [1,1,1]
    test2_vec = [2,3,4]
    test3_vec = [-2,6,17]
    test4_vec = [-6,-5,0]
    test5_vec = [-18, 9, 20]
    test6_vec = [-1,0,0]

    test1 = hpf.unit_vector_cart(test1_vec)
    test2 = hpf.unit_vector_cart(test2_vec)
    test3 = hpf.unit_vector_cart(test3_vec)
    test4 = hpf.unit_vector_cart(test4_vec)
    test5 = hpf.unit_vector_cart(test5_vec)
    test6 = hpf.unit_vector_cart(test6_vec)
    assert (test1 == [1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)]).all()
    assert (test2 == [2/np.sqrt(29),3/np.sqrt(29),1/np.sqrt(29)]).all()
    assert (test6 == [-1,0,0]).all()
    assert (test3 == [-2/np.sqrt(329),6/np.sqrt(329),17/np.sqrt(329)]).all()
    assert (test4 == [-6/np.sqrt(51),-5/np.sqrt(51),0]).all()
    assert (test5 == [-18/np.sqrt(805),9/np.sqrt(805),20/np.sqrt(805)]).all()




    

'''
def test_b_conversion():
'''