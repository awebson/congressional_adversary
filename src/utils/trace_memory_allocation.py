import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=10):
    """https://docs.python.org/3/library/tracemalloc.html"""

    # TODO write this as a context manager, then import as library
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),))
    top_stats = snapshot.statistics(key_type)

    print(f'Top {limit} lines')
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print(f'#{index}: {filename}:{frame.lineno}: '
              f'{stat.size / 1048576:.1f} MB')
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print(f'    {line}')

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other) / 1048576
        print(f"{len(other)} other: {size:.1f} MB")
    total = sum(stat.size for stat in top_stats) / 1048576
    print(f'Total allocated size: {total:.1f} MB')


# with torch.autograd.profiler.profile(use_cuda=True) as profile:
#     main()
# print(profile.key_averages().table(sort_by='cpu_time_total'))
# print(profile.key_averages().table(sort_by='cuda_time_total'))

# import linecache
# import os
# import tracemalloc

# def display_top(snapshot, key_type='lineno', limit=10):
#     snapshot = snapshot.filter_traces((
#         tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
#         tracemalloc.Filter(False, "<unknown>"),
#     ))
#     top_stats = snapshot.statistics(key_type)

#     print("Top %s lines" % limit)
#     for index, stat in enumerate(top_stats[:limit], 1):
#         frame = stat.traceback[0]
#         # replace "/path/to/module/file.py" with "module/file.py"
#         filename = os.sep.join(frame.filename.split(os.sep)[-2:])
#         print("#%s: %s:%s: %.1f KiB"
#             % (index, filename, frame.lineno, stat.size / 1024))
#         line = linecache.getline(frame.filename, frame.lineno).strip()
#         if line:
#             print('    %s' % line)

#     other = top_stats[limit:]
#     if other:
#         size = sum(stat.size for stat in other)
#         print("%s other: %.1f KiB" % (len(other), size / 1024))
#     total = sum(stat.size for stat in top_stats)
#     print("Total allocated size: %.1f KiB" % (total / 1024))

# tracemalloc.start()

# main()

# snapshot = tracemalloc.take_snapshot()
# display_top(snapshot)
