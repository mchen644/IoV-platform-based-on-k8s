package resources

type Profile struct {
	CpuCoresPercentage uint64 `json:"cpu_cores_percentage"`
	MemoryBytes        int64  `json:"memory_bytes"`
}

// NewResources
// cpuPercentage must between 0-100, represent use cpuPercentage% cpu resource
func NewResources(cpuPercentage uint64, memoryBytes int64) *Profile {
	return &Profile{
		CpuCoresPercentage: cpuPercentage,
		MemoryBytes:        memoryBytes,
	}
}
