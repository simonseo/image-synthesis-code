# aligning, stacking, post_processing algorithms
def dumb_stack(r,g,b):
    return np.dstack([r, g, b]), (0,0), (0,0)
def ssd_brute_force_stack(r,g,b):
    """
    Align channels by minimizing sum of squared differences.
    L2 norm also known as the Sum of Squared Differences (SSD) distance 
    which is simply sum(sum((image1-image2).^2)) 
    where the sum is taken over the pixel values.

    Args:
        r (np.ndarray): channel that is not aligned with the other two
        g (np.ndarray): channel that is not aligned with the other two
        b (np.ndarray): channel that is not aligned with the other two
    """
    def preprocess(channel):
        h, w = channel.shape
        channel = sk.transform.rescale(channel, 1/4) # scale down to reduce search space
        return channel

    def align(c1, c2):
        """align by ssd two channels"""
        scaled1 = preprocess(c1)
        scaled2 = preprocess(c2)
        min_ssd = np.Inf
        argmin_dx = 0
        argmin_dy = 0
        for dx in range(-10,10):
            for dy in range(-5,40):
                shifted = np.roll(scaled1, dx, axis=1)
                shifted = np.roll(shifted, dy, axis=0)
                ssd = np.sum(np.power(scaled2-shifted, 2))
                if ssd < min_ssd:
                    min_ssd = ssd
                    argmin_dx = dx
                    argmin_dy = dy
        c1 = np.roll(c1, argmin_dy*4, 0)
        c1 = np.roll(c1, argmin_dx*4, 1)
        return c1, (argmin_dy*4, argmin_dx*4)
    ag, g_shift = align(g, b)
    ar, r_shift = align(r, b)
    # show(np.dstack([ar, ag, b]))
    # show(dumb_stack(r,g,b))
    return np.dstack([ar, ag, b]), r_shift, g_shift

def ncc_brute_force_stack(r,g,b, crop=False):
    """
    normalized cross-correlation (NCC), which is simply a dot product 
    between two normalized vectors: (image1./||image1|| and image2./||image2||).

    Args:
        r (np.ndarray): channel that is not aligned with the other two
        g (np.ndarray): channel that is not aligned with the other two
        b (np.ndarray): channel that is not aligned with the other two
    """
    def preprocess(channel):
        h, w = channel.shape
        if crop:
            channel = channel[h*4.5//10:h-h*4.5//10, w*4.5//10:w-w*4.5//10] # only keep center 80%
        channel = sk.transform.rescale(channel, 1/4) # scale down to reduce search space
        channel = (channel-channel.mean())/channel.std() # normalizem
        return channel

    def align(c1, c2):
        """align by ssd two channels"""
        scaled1 = preprocess(c1)
        scaled2 = preprocess(c2)
        min_ncc = np.Inf
        argmin_dx = 0
        argmin_dy = 0
        for dx in range(-10,10):
            for dy in range(-5,40):
                shifted = np.roll(scaled1, dx, axis=1)
                shifted = np.roll(shifted, dy, axis=0)
                ncc = np.sum(np.power(scaled2-shifted, 2))
                if ncc < min_ncc:
                    min_ncc = ncc
                    argmin_dx = dx
                    argmin_dy = dy
        c1 = np.roll(c1, argmin_dy*4, 0)
        c1 = np.roll(c1, argmin_dx*4, 1)
        return c1, (argmin_dy*4, argmin_dx*4)
    ag, g_shift = align(g, b)
    ar, r_shift = align(r, b)
    return np.dstack([ar, ag, b]), r_shift, g_shift

#####################
# Iterative alignment using image pyramids
#####################


def ssd_align(c1, c2, center=(0,0)):
    """
    Align channels by minimizing sum of squared differences.
    L2 norm also known as the Sum of Squared Differences (SSD) distance 
    which is simply sum(sum((image1-image2).^2)) 
    where the sum is taken over the pixel values.
    """
    min_ssd = np.Inf
    argmin_dh = 0
    argmin_dw = 0
    for dh in range(-2,3):
        for dw in range(-2,3):
            shifted = np.roll(c1, center[0]+dh, axis=0)
            shifted = np.roll(shifted, center[1]+dw, axis=1)
            ssd = np.sum(np.power(c2-shifted, 2))
            if ssd > min_ssd:
                continue
            min_ssd = ssd
            argmin_dh = dh
            argmin_dw = dw
        print(f"{dh=} {min_ssd=}")
    return np.array((center[0]+argmin_dh, center[1]+argmin_dw))


def _ncc_align(c1, c2, center=(0,0), shift_fn=mirror_roll):
    """
    Align channels by minimizing sum of squared differences.
    L2 norm also known as the Sum of Squared Differences (SSD) distance 
    which is simply sum(sum((image1-image2).^2)) 
    where the sum is taken over the pixel values.

    shift_fn can be np.roll or mirror_roll.
    """

    def preprocess(channel):
        return channel/np.linalg.norm(channel)
        # return (channel-channel.mean())/channel.std() # normalizem

    c1 = preprocess(c1)
    c2 = preprocess(c2)
    min_ssd = np.Inf
    argmin_dh = 0
    argmin_dw = 0
    for dh in range(-2,3):
        for dw in range(-2,3):
            shifted = shift_fn(c1, center[0]+dh, axis=0)
            shifted = shift_fn(shifted, center[1]+dw, axis=1)
            ssd = np.sum(np.power(c2-shifted, 2))
            if ssd > min_ssd:
                continue
            min_ssd = ssd
            argmin_dh = dh
            argmin_dw = dw
        print(f"{dh=} {min_ssd=}")
    return np.array((center[0]+argmin_dh, center[1]+argmin_dw))

##################
# Create NCC functions
##################
ncc_align = partial(_ncc_align, roll=np.roll)
ncc_mirror_align = partial(_ncc_align, roll=mirror_roll)

##################
# Iterative alignment using image pyramids
##################
def iterative_align(r,g,b, align=ssd_align, depth=6):
    # depth = int(np.log2(np.min(r.shape)/10))
    # print(f"{depth=}")
    if depth == 0:
        return np.array([0,0]), np.array([0,0])

    # first align on smaller resolution / larger features
    half_r = sk.transform.rescale(r, 1/2)
    half_g = sk.transform.rescale(g, 1/2)
    half_b = sk.transform.rescale(b, 1/2)
    r_shift, g_shift = iterative_align(half_r, half_g, half_b, align, depth-1)

    # align on bigger resolution / smaller features
    r_shift = align(r,b, r_shift*2)
    g_shift = align(g,b, g_shift*2)
    print(f"{depth=} {r_shift=} {g_shift=}")

    return r_shift, g_shift
