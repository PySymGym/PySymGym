namespace ML.GameMaps.Simple;

public class ControlFlow
{
    public static int BinarySearch(int[] a, int x, int lo, int hi)
    {
        if (a == null) throw new ArgumentException("a == null");

        if (lo < 0) throw new ArgumentException("lo < 0");
        if (lo > hi) throw new ArgumentException("lo > hi");

        var m = lo + (hi - lo) / 2;

        while (lo < hi)
            if (a[m] == x)
                return m;
            else if (a[m] > x)
                hi = m;
            else
                lo = m + 1;

        return -1;
    }

    public static int Switches1(int x, int y, int z)
    {
        var sum = 0;
        switch (x % 10)
        {
            case 0:
                sum += 1;
                break;
            case 1:
                sum += 1;
                break;
            case 2:
                sum += 1;
                break;
            case 3:
                sum += 1;
                break;
            case 4:
                sum += 1;
                break;
            case 5:
                sum += 2;
                break;
            case 6:
                sum += 2;
                break;
            case 7:
                sum += 2;
                break;
            case 8:
                sum += 2;
                break;
            case 9:
                sum += 2;
                break;
        }

        switch (y % 10)
        {
            case 0:
                sum += 1;
                break;
            case 1:
                sum += 1;
                break;
            case 2:
                sum += 1;
                break;
            case 3:
                sum += 1;
                break;
            case 4:
                sum += 1;
                break;
            case 5:
                sum += 2;
                break;
            case 6:
                sum += 2;
                break;
            case 7:
                sum += 2;
                break;
            case 8:
                sum += 2;
                break;
            case 9:
                sum += 2;
                break;
        }

        switch (z % 10)
        {
            case 0:
                sum += 1;
                break;
            case 1:
                sum += 1;
                break;
            case 2:
                sum += 1;
                break;
            case 3:
                sum += 1;
                break;
            case 4:
                sum += 1;
                break;
            case 5:
                sum += 2;
                break;
            case 6:
                sum += 2;
                break;
            case 7:
                sum += 2;
                break;
            case 8:
                sum += 2;
                break;
            case 9:
                sum += 2;
                break;
        }

        return sum;
    }

    public static string Switches2(int x)
    {
        List<string> numbers = new List<string>();
        int val = x;
        while (val != 0)
        {
            switch (val % 10)
            {
                case 0:
                    numbers.Add("Zero");
                    break;
                case 1:
                    numbers.Add("One");
                    break;
                case 2:
                    numbers.Add("Two");
                    break;
                case 3:
                    numbers.Add("Three");
                    break;
                case 4:
                    numbers.Add("Four");
                    break;
                case 5:
                    numbers.Add("Five");
                    break;
                case 6:
                    numbers.Add("Six");
                    break;
                case 7:
                    numbers.Add("Seven");
                    break;
                case 8:
                    numbers.Add("Eight");
                    break;
                case 9:
                    numbers.Add("Nine");
                    break;
            }

            numbers.Add("; ");
            val = val / 10;
        }

        return String.Concat(numbers);
    }

    public List<int> Switches3(List<string> x)
    {
        List<int> result = new List<int>();
        foreach (var name in x)
        {
            switch (name)
            {
                case "zero":
                    result.Add(0);
                    break;
                case "one":
                    result.Add(1);
                    break;
                case "two":
                    result.Add(2);
                    break;
                case "three":
                    result.Add(3);
                    break;
                case "four":
                    result.Add(4);
                    break;
                case "five":
                    result.Add(5);
                    break;
                case "six":
                    result.Add(6);
                    break;
                case "seven":
                    result.Add(7);
                    break;
                case "eight":
                    result.Add(8);
                    break;
                case "nine":
                    result.Add(9);
                    break;
            }
        }

        return result;
    }


    public List<int> Switches3_1(List<string> x)
    {
        List<int> result = new List<int>();
        foreach (var name in x)
        {
            switch (name)
            {
                case "a":
                    result.Add(0);
                    break;
                case "b":
                    result.Add(1);
                    break;
                default:
                    result.Add(2);
                    break;
            }
        }

        return result;
    }

    public List<int> Switches3_2(List<string> x)
    {
        List<int> result = new List<int>();
        foreach (var name in x)
        {
            switch (name)
            {
                case "a":
                    result.Add(0);
                    break;
                case "b":
                    result.Add(1);
                    break;
                case "c":
                    result.Add(2);
                    break;
                case "d":
                    result.Add(3);
                    break;
                case "e":
                    result.Add(4);
                    break;
                default:
                    result.Add(5);
                    break;
            }
        }
        return result;
    }

    public List<int> Switches3_3(List<string> x)
    {
        List<int> result = new List<int>();
        foreach (var name in x)
        {
            switch (name)
            {
                case "a":
                    result.Add(0);
                    break;
                case "b":
                    result.Add(1);
                    break;
                case "c":
                    result.Add(2);
                    break;
                case "d":
                    result.Add(3);
                    break;
                case "e":
                    result.Add(4);
                    break;
                case "f":
                    result.Add(5);
                    break;
            }
        }

        return result;
    }


    public List<string> Switches6(List<int> x, int i, List<string> res)
    {        
            if (i < x.Count - 1)
            {
            switch (x[i])
            {
                case 0:
                    switch (x[i+1])
                    {
                        case 0:
                            res.Add("zero");
                            Switches6(x, i + 2, res);
                            break;
                        case 1:
                            res.Add("one");
                            Switches6(x, i + 2, res);
                            break;
                        default:
                            Switches6(x, i + 1, res);
                            break;
                    }
                    break;
                case 1:
                    switch (x[i+1])
                    {
                        case 0:
                            res.Add("ten");
                            Switches6(x, i + 2, res);
                            break;
                        case 1:
                            res.Add("eleven");
                            Switches6(x, i + 2, res);
                            break;
                        default:
                            Switches6(x, i + 1, res);
                            break;
                    }
                    break;
                case 2:
                    switch (x[i+1])
                    {
                        case 0:
                            res.Add("twenty");
                            Switches6(x, i + 2, res);
                            break;
                        case 1:
                            res.Add("twenty-one");
                            Switches6(x, i + 2, res);
                            break;
                        default:
                            Switches6(x, i + 1, res);
                            break;
                    }
                    break;
                case 3:
                    switch (x[i+1])
                    {
                        case 0:
                            res.Add("30");
                            Switches6(x, i + 2, res);
                            break;
                        case 1:
                            res.Add("31");
                            Switches6(x, i + 2, res);
                            break;
                        case 2:
                            res.Add("32");
                            Switches6(x, i + 2, res);
                            break;
                        case 3:
                            res.Add("33");
                            Switches6(x, i + 2, res);
                            break;
                        default:
                            Switches6(x, i + 1, res);
                            break;
                    }
                    break;
                case 4:
                    switch (x[i+1])
                    {
                        case 0:
                            res.Add("40");
                            Switches6(x, i + 2, res);
                            break;
                        case 1:
                            res.Add("41");
                            Switches6(x, i + 2, res);
                            break;
                        case 2:
                            res.Add("42");
                            Switches6(x, i + 2, res);
                            break;
                        case 3:
                            res.Add("43");
                            Switches6(x, i + 2, res);
                            break;
                        case 4:
                            res.Add("44");
                            Switches6(x, i + 2, res);
                            break;
                        default:
                            Switches6(x, i + 1, res);
                            break;
                    }
                    break;
                case 5:
                    res.Add("five");
                    Switches6(x, i + 1, res);
                    break;
                case 6:
                    res.Add("six");
                    Switches6(x, i + 1, res);
                    break;
                case 7:
                    res.Add("seven");
                    Switches6(x, i + 1, res);
                    break;
                case 8:
                    res.Add("eight");
                    Switches6(x, i + 1, res);
                    break;
                case 9:
                    res.Add("nine");
                    Switches6(x, i + 1, res);
                    break;
                default:
                    res.Add("unexpected");
                    Switches6(x, i + 1, res);
                    break;
            }
            
        }
        return res;
     
    }

    public List<string> Switches4_1(List<int> x)
    {
        List<string> result = new List<string>();
        foreach (var name in x)
        {
            switch (name)
            {
                case 0:
                    result.Add("0");
                    break;
                case 1:
                    result.Add("1");
                    break;
                default:
                    result.Add("-1");
                    break;                
            }
        }

        return result;
    }

    public List<string> Switches4_2(List<int> x)
    {
        List<string> result = new List<string>();
        foreach (var name in x)
        {
            switch (name)
            {
                case 0:
                    result.Add("0");
                    break;
                case 1:
                    result.Add("1");
                    break;
                case 2:
                    result.Add("2");
                    break;
                case 3:
                    result.Add("3");
                    break;
                default:
                    result.Add("-1");
                    break;                
            }
        }

        return result;
    }

    public List<string> Switches4_3(List<int> x)
    {
        List<string> result = new List<string>();
        foreach (var name in x)
        {
            switch (name)
            {
                case 0:
                    result.Add("0");
                    break;
                case 1:
                    result.Add("1");
                    break;
                case 2:
                    result.Add("2");
                    break;
                case 3:
                    result.Add("3");
                    break;
                case 4:
                    result.Add("4");
                    break;
                case 5:
                    result.Add("5");
                    break;

                default:
                    result.Add("-1");
                    break;                
            }
        }

        return result;
    }


    public List<string> Switches4(List<int> x)
    {
        List<string> result = new List<string>();
        foreach (var name in x)
        {
            switch (name)
            {
                case 0:
                    result.Add("0");
                    break;
                case 1:
                    result.Add("1");
                    break;
                case 2:
                    result.Add("2");
                    break;
                case 3:
                    result.Add("3");
                    break;
                case 4:
                    result.Add("4");
                    break;
                case 5:
                    result.Add("5");
                    break;
                case 6:
                    result.Add("6");
                    break;
                case 7:
                    result.Add("7");
                    break;
                case 8:
                    result.Add("8");
                    break;
                case 9:
                    result.Add("9");
                    break;
            }
        }

        return result;
    }

    public bool Switches5(List<int> x)
    {
        if (x.Count == 0)
        {
            return false;
        }

        switch (x[0])
        {
            case 0:
                return true;
            case 1:
                return (Switches5(x.Skip(1).ToList()));
            case 2:
                return (Switches5(x.Skip(2).ToList()));
            default:
                return (Switches5(x.Skip(3).ToList()));
        }
    }

    public static int NestedFors(int x)
    {
        int sum = 0;
        for (int i = 1; i <= x; i++)
        {
            for (int j = 1; j <= i; j++)
            {
                for (int k = 1; k <= j; k++)
                {
                    sum += k;
                }
            }
        }

        return sum;
    }
}