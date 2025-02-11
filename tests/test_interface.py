from pangolin.interface import *


def test_constant1():
    x = Constant(0)
    assert x.get_shape() == ()


def test_constant2():
    x = Constant([0, 1, 2])
    assert x.get_shape() == (3,)


def test_normal1():
    x = normal(0, 1)
    assert x.cond_dist == normal_scale
    assert x.shape == ()
    assert x.ndim == 0


def test_normal2():
    x = normal(0, scale=1)
    assert x.cond_dist == normal_scale
    assert x.shape == ()
    assert x.ndim == 0


def test_normal3():
    x = normal(0, prec=1)
    assert x.cond_dist == normal_prec
    assert x.shape == ()
    assert x.ndim == 0


def test_normal4():
    try:
        # should fail because can't provide both scale and prec
        x = normal(0, scale=1, prec=1)
        assert False
    except Exception as e:
        assert True


def test_tform1():
    x = normal(0, 1)
    y = x * x + x
    assert y.shape == ()
    assert y.ndim == 0


def test_tform2():
    x = normal(0, 1)
    y = normal(0, 1)
    z = x * y + y ** (y**2)
    assert z.shape == ()
    assert z.ndim == 0


def test_VMapDist1():
    # couldn't call normal here because it's not a CondDist. But I guess that's fine because user isn't expected
    # to touch VMap directly
    diag_normal = VMapDist(normal_scale, [0, 0], 3)
    assert diag_normal.get_shape((3,), (3,)) == (3,)


def test_VMapDist2():
    diag_normal = VMapDist(normal_scale, [0, None], 3)
    assert diag_normal.get_shape((3,), ()) == (3,)


def test_VMapDist3():
    diag_normal = VMapDist(normal_scale, [None, 0], 3)
    assert diag_normal.get_shape((), (3,)) == (3,)


def test_VMapDist4():
    diag_normal = VMapDist(normal_scale, [None, None], 3)
    assert diag_normal.get_shape((), ()) == (3,)


def test_VMapDist5():
    diag_normal = VMapDist(normal_scale, [None, 0], 3)
    try:
        # should fail because shapes are incoherent
        x = diag_normal.get_shape((), (4,))
        assert False
    except AssertionError as e:
        assert True


def test_double_VMapDist1():
    # couldn't call normal here because it's not a CondDist. But I guess that's fine because user isn't expected
    # to touch VMap directly
    diag_normal = VMapDist(normal_scale, [0, 0])
    matrix_normal = VMapDist(diag_normal, [0, 0])
    assert matrix_normal.get_shape((4, 3), (4, 3)) == (4, 3)


def test_vmap_dummy_args1():
    a = makerv(np.ones((5, 3)))
    b = makerv(np.ones(5))
    (dummy_a, dummy_b), axis_size = vmap_dummy_args([0, 0], 5, a, b)
    assert dummy_a.shape == (3,)
    assert dummy_b.shape == ()
    assert axis_size == 5


def test_vmap_dummy_args2():
    a = makerv(np.ones((5, 3)))
    b = makerv(np.ones((3, 5)))
    (dummy_a, dummy_b), axis_size = vmap_dummy_args([0, 1], 5, a, b)
    assert dummy_a.shape == (3,)
    assert dummy_b.shape == (3,)
    assert axis_size == 5


def test_vmap_dummy_args3():
    a = makerv(np.ones((5, 3, 2)))
    b = makerv(np.ones((3, 1, 9)))
    (dummy_a, dummy_b), axis_size = vmap_dummy_args([1, None], None, a, b)
    assert dummy_a.shape == (5, 2)
    assert dummy_b.shape == (3, 1, 9)
    assert axis_size == 3


def test_vmap_generated_nodes1():
    # both inputs explicitly given
    a = AbstractRV(())
    b = AbstractRV(())
    f = lambda a, b: [normal(a, b)]
    nodes = vmap_generated_nodes(f, a, b)[0]
    assert len(nodes) == 1
    assert nodes[0].cond_dist == normal_scale


def test_vmap_generated_nodes2():
    # b captured as a closure
    a = AbstractRV(())
    b = AbstractRV(())
    f = lambda a: [normal(a, b)]
    nodes = vmap_generated_nodes(f, a)[0]
    assert len(nodes) == 1
    assert nodes[0].cond_dist == normal_scale


def test_vmap_generated_nodes3():
    # a captured as a closure
    a = AbstractRV(())
    b = AbstractRV(())
    f = lambda b: [normal(a, b)]
    nodes = vmap_generated_nodes(f, b)[0]
    assert len(nodes) == 1
    assert nodes[0].cond_dist == normal_scale


def test_vmap_generated_nodes4():
    # both a and b captured as a closure
    a = AbstractRV(())
    b = AbstractRV(())
    f = lambda: [normal(a, b)]
    nodes = vmap_generated_nodes(f)[0]
    assert len(nodes) == 1
    assert nodes[0].cond_dist == normal_scale


def test_vmap_generated_nodes5():
    def fun(a, b):
        loc = a + b
        scale = 1
        return [normal(loc, scale)]

    a = AbstractRV(())
    b = AbstractRV(())

    # both a and b given
    f = lambda a, b: fun(a, b)
    nodes = list(vmap_generated_nodes(f, a, b)[0])
    assert len(nodes) == 3
    assert nodes[0].cond_dist == add
    assert isinstance(nodes[1].cond_dist, Constant)
    assert nodes[2].cond_dist == normal_scale

    # b captured with closure
    f = lambda a: fun(a, b)
    nodes = list(vmap_generated_nodes(f, a)[0])
    assert len(nodes) == 3
    assert nodes[0].cond_dist == add
    assert isinstance(nodes[1].cond_dist, Constant)
    assert nodes[2].cond_dist == normal_scale

    # neither a nor b captured
    f = lambda: fun(a, b)
    nodes = list(vmap_generated_nodes(f)[0])
    assert len(nodes) == 3
    assert nodes[0].cond_dist == add
    assert isinstance(nodes[1].cond_dist, Constant)
    assert nodes[2].cond_dist == normal_scale


def test_vmap_eval1():
    "should fail because of incoherent axes sizes"
    try:
        y = vmap_eval(
            lambda loc, scale: normal_scale(loc, scale),
            [None, None],
            5,
            np.zeros(3),
            np.ones(3),
        )
        assert False
    except AssertionError as e:
        assert True


def test_vmap_eval2():
    y = vmap_eval(
        lambda loc, scale: [normal_scale(loc, scale)], [0, None], 3, np.zeros(3), 1
    )[0]
    assert y.shape == (3,)


def test_vmap_eval3():
    def f(x):
        return [normal(x, x)]

    y = vmap_eval(f, [0], None, np.array([3.3, 4.4, 5.5]))[0]
    assert y.shape == (3,)


def test_vmap_eval4():
    def f(x):
        loc = x * 1.1
        scale = x**2.2
        return [normal(loc, scale)]

    y = vmap_eval(f, [0], None, np.array([3.3, 4.4, 5.5]))[0]
    assert y.shape == (3,)


def test_vmap_eval6():
    def f():
        return [normal(0, 1)]

    x = vmap_eval(f, [], 3)[0]
    assert x.shape == (3,)


def test_vmap_eval7():
    def f(x):
        loc = x * 1.1
        scale = x**2.2
        y = normal(loc, scale)
        x = normal(0, 1)
        z = normal(1, 2)
        return [y, x, z]

    y, x, z = vmap_eval(f, [0], 3, np.array([3.3, 4.4, 5.5]))
    assert y.shape == (3,)
    assert x.shape == (3,)
    assert z.shape == (3,)


def test_vmap_eval8():
    def f(x):
        loc = x * 1.1
        scale = x**2.2
        y = normal(loc, scale)
        x = normal(0, 1)
        z = normal(1, 2)
        return [y, x, z]

    y, x, z = vmap_eval(f, [0], None, np.array([3.3, 4.4, 5.5]))
    assert y.shape == (3,)
    assert x.shape == (3,)
    assert z.shape == (3,)


def test_vmap1():
    y = vmap(normal_scale, (0, None), 3)(np.zeros(3), np.array(1))
    assert y.shape == (3,)


def test_vmap2():
    def f(x):
        return normal(x, x)

    y = vmap(f, in_axes=0, axis_size=3)(np.array([3.3, 4.4, 5.5]))
    assert y.shape == (3,)


def test_vmap3():
    def f(x):
        loc = x * 1.1
        scale = x**2.2
        return normal(loc, scale)

    y = vmap(f, in_axes=0, axis_size=None)(np.array([3.3, 4.4, 5.5]))
    assert y.shape == (3,)


def test_vmap4():
    def f():
        return normal(np.array(1), np.array(2))

    y = vmap(f, in_axes=None, axis_size=3)()
    assert y.shape == (3,)


def test_vmap5():
    def f(loc, scale):
        x = normal(loc, scale)
        y = normal(x, scale)
        return (x, y)

    loc = np.array([1, 2, 3])
    scale = np.array([4, 5, 6])
    x, y = vmap(f, in_axes=(0, 0), axis_size=3)(loc, scale)
    assert x.shape == (3,)
    assert y.shape == (3,)


def test_vmap6():
    def f(loc, scale):
        x = normal(loc, scale)
        y = normal(x, scale)
        return (x, y)

    loc = np.array([1, 2, 3])
    scale = np.array([4, 5, 6])
    x, y = vmap(f, in_axes=(0, 0), axis_size=None)(loc, scale)
    assert x.shape == (3,)
    assert y.shape == (3,)


def test_vmap7():
    def f(loc, scale):
        x = normal(loc, scale)
        y = normal(x, scale)
        return (x, y)

    loc = np.array([1, 2, 3])
    scale = np.array([4, 5, 6])
    x, y = vmap(f, in_axes=0, axis_size=None)(loc, scale)
    assert x.shape == (3,)
    assert y.shape == (3,)


def test_vmap8():
    def f(stuff):
        x = stuff["x"]
        y, z = stuff["yz"]
        a = normal(x, 1)
        b = normal(a, y)
        c = normal(b, z)
        return ({"a": a}, b, c)

    stuff = {"x": 1.1, "yz": (2.2, 3.3)}
    out = vmap(f, in_axes=None, axis_size=5)(stuff)
    assert out[0]["a"].shape == (5,)
    assert out[1].shape == (5,)
    assert out[2].shape == (5,)


def test_vmap9():
    def f(stuff):
        x = stuff["x"]
        y, z = stuff["yz"]
        a = normal(x, 1)
        b = normal(a, y)
        c = normal(b, z)
        return ({"a": a}, b, c)

    stuff = {"x": 1.1, "yz": (2.2 * np.ones(5), 3.3)}
    # this doesn't work, for unclear jax reasons
    # in_axes = {'x': None, 'yz': (0, None)}
    # but this does
    in_axes = ({"x": None, "yz": (0, None)},)
    out = vmap(f, in_axes=in_axes, axis_size=5)(stuff)
    assert out[0]["a"].shape == (5,)
    assert out[1].shape == (5,)
    assert out[2].shape == (5,)


def test_vmap10():
    def f(x):
        loc = x * 1.1
        scale = x**2.2
        return {"y": normal(loc, scale)}

    stuff = vmap(f, 0, None)(np.array([3.3, 4.4, 5.5]))
    assert stuff["y"].shape == (3,)


def test_vmap11():
    def f(x):
        loc = x * 1.1
        scale = x**2.2
        y = normal(loc, scale)
        x = normal(0, 1)
        z = normal(1, 2)
        return {"y": y, "xz": (x, z)}

    stuff = vmap(f, 0, None)(np.array([3.3, 4.4, 5.5]))
    # fancy pattern matching
    match stuff:
        case {"y": y, "xz": (x, z)}:
            assert y.shape == (3,)
            assert x.shape == (3,)
            assert z.shape == (3,)
        case _:
            assert False, "should be impossible"


def test_vmap12():
    loc = 0.5

    def f(scale):
        return normal(loc, scale)

    x = vmap(f, 0, None)(np.array([2.2, 3.3, 4.4]))
    assert x.shape == (3,)


def test_vmap13():
    loc = 0.5
    scale = 1.3

    def f():
        return normal(loc, scale)

    x = vmap(f, None, 3)()
    assert x.shape == (3,)


def test_vmap14():
    x = normal(1.1, 2.2)
    y = vmap(lambda: normal(x, 1), None, 3)()
    assert y.shape == (3,)


def test_vmap15():
    x = normal(0, 1)
    y, z = vmap(
        lambda: (yi := normal(x, 2), zi := vmap(lambda: normal(yi, 3), None, 5)()),
        None,
        3,
    )()


def test_plate1():
    x = normal(0, 1)
    y = plate(N=3)(lambda: normal(x, 1))
    assert x.shape == ()
    assert y.shape == (3,)


def test_plate2():
    x = normal(0, 1)
    y, z = plate(N=3)(
        lambda: (yi := normal(x, 1), zi := plate(N=5)(lambda: normal(yi, 1)))
    )
    assert x.shape == ()
    assert y.shape == (3,)
    assert z.shape == (3, 5)


def test_plate3():
    x = normal(0, 1)
    y = plate(N=3)(lambda: normal(x, 1))
    z = plate(y, N=3)(lambda yi: plate(N=5)(lambda: normal(yi, 1)))
    assert x.shape == ()
    assert y.shape == (3,)
    assert z.shape == (3, 5)


def test_plate3a():
    x = normal(0, 1)
    y = plate(N=3)(lambda: normal(x, 1))
    z = plate(y)(lambda yi: plate(N=5)(lambda: normal(yi, 1)))
    assert x.shape == ()
    assert y.shape == (3,)
    assert z.shape == (3, 5)


def test_plate4():
    x = normal(0, 1)
    y = plate(N=3)(lambda: normal(x, 1))
    z = plate(y, N=3)(lambda yi: normal(yi, 1))
    assert x.shape == ()
    assert y.shape == (3,)
    assert z.shape == (3,)


def test_plate5():
    loc = np.array([2, 3, 4])
    scale = np.array([5, 6, 7])
    x = plate(loc, scale, N=3)(normal)
    assert x.shape == (3,)


def test_plate6():
    loc = np.array([2, 3, 4])
    scale = np.array(5)
    x = plate(loc, scale, N=3, in_axes=(0, None))(normal)
    assert x.shape == (3,)


def test_plate6a():
    "recommended implementation"
    loc = np.array([2, 3, 4])
    scale = np.array(5)
    x = plate(loc)(lambda loc_i: normal(loc_i, scale))
    assert x.shape == (3,)


def test_plate7():
    "not recommended but legal"  # recommended implmentation same as 6a
    loc = np.array([2, 3, 4])
    scale = np.array(5)
    x = plate(loc, scale, N=None, in_axes=(0, None))(normal)
    assert x.shape == (3,)


def test_plate8():
    loc = np.array(2)
    scale = np.array(5)
    x = plate(loc, scale, N=3, in_axes=(None, None))(normal)
    assert x.shape == (3,)


def test_plate8a():
    loc = np.array(2)
    scale = np.array(5)
    x = plate(N=3)(lambda: normal(loc, scale))
    assert x.shape == (3,)


def test_plate9():
    loc = np.array(2)
    scale = np.array(5)
    x = plate(loc, scale, N=3, in_axes=None)(normal)
    assert x.shape == (3,)


def test_plate10():
    loc = np.array(2)
    scale = np.array(5)
    x = plate(N=3)(lambda: normal(loc, scale))
    assert x.shape == (3,)


def test_plate11():
    x = plate(N=5)(lambda: normal(0, 1))
    y = plate(N=3)(lambda: normal(0, 1))
    z = plate(x)(lambda xi: plate(y)(lambda yi: normal(xi * yi, 1)))
    assert x.shape == (5,)
    assert y.shape == (3,)
    assert z.shape == (5, 3)


def test_plate12():
    stuff = plate(N=5)(lambda: {"x": normal(0, 1)})
    assert stuff["x"].shape == (5,)


def test_index1():
    d = Index(None, None)
    assert d.get_shape((3, 2), (), ()) == ()


def test_index2():
    d = Index(None, None)
    assert d.get_shape((3, 2), (4,), (4,)) == (4,)


def test_index3():
    d = Index(slice(None), None)
    assert d.get_shape((3, 2), (4,)) == (3, 4)


def test_index4():
    d = Index(None, slice(None))
    assert d.get_shape((3, 2), (4,)) == (4, 2)


def test_index5():
    d = Index(None, slice(None))
    assert d.get_shape((3, 2), (4, 5, 7)) == (4, 5, 7, 2)


def test_indexing1():
    # TODO: make this and all the following tests directly test against numpy functionality
    x = makerv([1, 2, 3])
    y = x[0]
    assert isinstance(y.cond_dist, Index)
    assert y.shape == ()


def test_indexing2():
    x = makerv([1, 2, 3])
    idx = makerv([0, 1])
    y = x[idx]
    assert isinstance(y.cond_dist, Index)
    assert y.shape == (2,)


def test_indexing3():
    x = makerv([1, 2, 3])
    idx = [0, 1]
    y = x[idx]
    assert isinstance(y.cond_dist, Index)
    assert y.shape == (2,)


def test_indexing4():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    idx = [0, 1, 1, 0]
    y = x[idx, :]
    assert isinstance(y.cond_dist, Index)
    assert y.shape == (4, 3)


def test_indexing5():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    idx = [[0, 1], [1, 1], [0, 0], [1, 0]]
    y = x[:, idx]
    assert isinstance(y.cond_dist, Index)
    assert y.shape == (2, 4, 2)


def test_indexing6():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    idx = [0, 1, 1, 0]
    y = x[idx]
    assert isinstance(y.cond_dist, Index)
    assert y.shape == (4, 3)


# def test_indexing7():
#     x_numpy = np.random.randn(5, 6, 7, 8, 9)
#     y_numpy = x[:, [1, 2, 3]]
