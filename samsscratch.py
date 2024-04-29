# detect tags

# Tag corners must be no less than this number of pixels from the image border
buffer_px = 10

# Iterate over all images in the source directory
for view in views:
    # Read image as grayscale
    img = view['img']

    # Detect tags
    tag_detections = tag_detector.detect(
        img,
        estimate_tag_pose=True,
        camera_params=np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]),
        tag_size=0.024,
    )

    rejected_tags = []
    tagdata = []
    p = []
    q = []
    for d in tag_detections:

        # Reject tags with corners too close to the image boundary
        if ((d.corners[:, 0] < buffer_px).any() or
                (d.corners[:, 0] > (img.shape[1] - 1) - buffer_px).any() or
                (d.corners[:, 1] < buffer_px).any() or
                (d.corners[:, 1] > (img.shape[0] - 1) - buffer_px).any()):
            continue

        # Add each point
        for c in d.corners:
            view['pts'] = [
                {
                    'pt2d': np.array(c),
                    'track': None,
                }
            ]
            view['desc'] = str(tag["tag_id"]) + "2")

            tagdata.append({
                'tag_id': d.tag_id,
                'corners': d.corners.tolist(),
                'center': d.center,
                'R': d.pose_R,
                'p': d.pose_t,
                'error': d.pose_err,
            })

            # Add corners of tag to point correspondences
            p.extend(get_tag_with_id(d.tag_id, template)['corners'])
            q.extend(d.corners.tolist())

            view['tags'] = tags

            # Make sure the lengths of p and q are consistent
            assert (len(p) == len(q))

            # Count the number of tags and correspondences that were found
            num_tags = len(tags)
            num_points = len(p)

            ######
            pts, desc = sift.detectAndCompute(image=view['img'], mask=None)
            view['pts'] = [
        {
            'pt2d': np.array(pt.pt),
            'track': None,
        }
        for pt in pts
    ]
    view['desc'] = desc