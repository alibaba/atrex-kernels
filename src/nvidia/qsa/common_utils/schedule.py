"""Cost ordering in four-block neighborhoods preserves short-range locality."""
import triton
import triton.language as tl


@triton.jit
def atrex_qsa_reset_ticket(ticket):
    """Start a call-local prefill ticket epoch; included in public-call timing."""
    tl.store(ticket, 0)


@triton.jit
def atrex_qsa_order(COUNT,N:tl.constexpr,T:tl.constexpr,PDL:tl.constexpr):
    # Only the downstream attention is a PDL-dependent launch. This sort
    # runs after the preprocessor has fully completed on the same stream.
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    group=tl.arange(0,T//4)
    offset=tl.arange(0,4)
    base=tl.program_id(0)*T
    block=base+group[:,None]*4+offset[None,:]
    count=tl.load(COUNT+block,block<N,other=0)
    cost=tl.sum(count,1)
    key=tl.where(base+group*4<N,cost*(T//4)+group,-1)
    ordered=tl.sort(key,descending=True)%(T//4)
    index=tl.reshape(ordered[:,None]*4+(3-offset[None,:]),(T,))
    valid=(index>=0)&(base+index<N)
    compact=tl.cumsum(valid.to(tl.int32))-1
    tl.store(COUNT+N+base+compact,base+index,valid)


def schedule(count,n,pdl):
    # Bound both the single-program sort and its int32 cost/tie-break key.
    # The established <=4096-block schedule is unchanged; larger launches
    # concatenate complete, independently ordered segments.
    tile=min(4096,max(4,triton.next_power_of_2(n)))
    atrex_qsa_order[(triton.cdiv(n,tile),)](count,n,tile,pdl,num_warps=8)
