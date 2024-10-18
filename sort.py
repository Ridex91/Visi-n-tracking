
        # Continúa el código de Sort...
 
    def update(self, dets=np.empty((0,6)), unique_color=False):
        self.frame_count += 1
        
        # Obtener ubicaciones predichas de los rastreadores existentes
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            if unique_color:
                self.color_list.pop(t)
        
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        
        # Actualizar rastreadores emparejados con las detecciones asignadas
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
        
        # Crear e inicializar nuevos rastreadores para detecciones no emparejadas
        for i in unmatched_dets:
            trk = KalmanBoxTracker(np.hstack((dets[i, :], np.array([0]))))
            self.trackers.append(trk)
            if unique_color:
                self.color_list.append(get_color())
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1))
            i -= 1
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                if unique_color:
                    self.color_list.pop(i)

        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0,6))


def parse_args():
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]', action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", help="Maximum number of frames to keep alive a track without associated detections.", type=int, default=1)
    parser.add_argument("--min_hits", help="Minimum number of associated detections before track is initialised.", type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)

    if(display):
        if not os.path.exists('mot_benchmark'):
            print('\n\tERROR: mot_benchmark link not found!\n\n')
        exit()

    if not os.path.exists('output'):
        os.makedirs('output')

    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    for seq_dets_fn in glob.glob(pattern):
        mot_tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    
    # Aquí continuaría el procesamiento de frames en el caso de display y output...
